from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from lfrm.config import ModelConfig, RuntimeConfig
from .common import Array, casted_linear_init, compute_dtype, maybe_cast, trunc_normal_init
from .recurrent.layers import (
    dot_product_attention,
    multi_head_attention_with_rope,
    swiglu_intermediate_size,
    unscaled_rms_norm,
)


class BRCLocalConvSwiGLU(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        hidden_dim: int,
        mlp_ratio: int,
        dtype: jnp.dtype,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        brc = config.brc_config
        if brc.local_kernel < 1 or brc.local_kernel % 2 == 0:
            raise ValueError("BRC local_kernel must be a positive odd integer")
        self.config = config
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        intermediate_size = swiglu_intermediate_size(hidden_dim, mlp_ratio, min_size=hidden_dim)
        self.gate_up = nnx.Linear(
            hidden_dim,
            2 * intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.depthwise = nnx.Conv(
            intermediate_size,
            intermediate_size,
            kernel_size=(brc.local_kernel, brc.local_kernel),
            padding="SAME",
            feature_group_count=intermediate_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=jnp.float32,
            preferred_element_type=dtype,
            rngs=rngs,
        )
        self.down = nnx.Linear(
            intermediate_size,
            hidden_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )

    def __call__(self, h: Array) -> Array:
        batch_size = h.shape[0]
        gate, up = jnp.split(self.gate_up(maybe_cast(h, self.dtype)), 2, axis=-1)
        x = jax.nn.silu(gate) * up
        x = x.reshape(
            batch_size,
            self.config.grid_height,
            self.config.grid_width,
            x.shape[-1],
        )
        x = self.depthwise(x)
        x = x.reshape(batch_size, self.config.seq_len, x.shape[-1])
        return self.down(jax.nn.silu(x))


class BRCSolverBlock(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: int,
        dtype: jnp.dtype,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.dtype = dtype
        self.self_attention = nnx.MultiHeadAttention(
            num_heads,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            dtype=dtype,
            param_dtype=jnp.float32,
            use_bias=False,
            dropout_rate=0.0,
            attention_fn=dot_product_attention,
            kernel_init=casted_linear_init,
            out_kernel_init=casted_linear_init,
            rngs=rngs,
        )
        brc = config.brc_config
        self.attn_norm = unscaled_rms_norm(hidden_dim, brc.rms_norm_eps, dtype, rngs)
        self.attn_context_norm = unscaled_rms_norm(hidden_dim, brc.rms_norm_eps, dtype, rngs)
        self.local_norm = unscaled_rms_norm(hidden_dim, brc.rms_norm_eps, dtype, rngs)
        self.local_context_norm = unscaled_rms_norm(hidden_dim, brc.rms_norm_eps, dtype, rngs)
        self.attn_scale = float(brc.attn_scale)
        self.local_scale = float(brc.local_scale)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.local_mlp = BRCLocalConvSwiGLU(
            config,
            hidden_dim,
            mlp_ratio,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        h: Array,
        context_condition: Array,
        trajectory_condition: Array,
        rope_cos: Array | None,
        rope_sin: Array | None,
    ) -> Array:
        h = self.global_communicate(
            h,
            context_condition,
            trajectory_condition,
            rope_cos,
            rope_sin,
        )
        return self.local_think(h, context_condition, trajectory_condition)

    def global_communicate(
        self,
        h: Array,
        context_condition: Array,
        trajectory_condition: Array,
        rope_cos: Array | None,
        rope_sin: Array | None,
    ) -> Array:
        route_condition = self.attn_context_norm(context_condition.astype(jnp.float32)).astype(jnp.float32)
        q_hint = trajectory_condition.astype(jnp.float32)
        # Attention routing is anchored by the hidden workspace and hard context.
        # The stopped q/history signal is only a value hint for energy-context construction.
        route = (
            self.attn_norm(h.astype(jnp.float32)).astype(jnp.float32)
            + route_condition
        ).astype(self.dtype)
        value = (route.astype(jnp.float32) + q_hint).astype(self.dtype)
        attn = multi_head_attention_with_rope(
            self.self_attention,
            route,
            route,
            value,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            name="BRC energy-context attention",
        )
        return (h.astype(jnp.float32) + self.attn_scale * attn.astype(jnp.float32)).astype(self.dtype)

    def local_think(
        self,
        h: Array,
        context_condition: Array,
        trajectory_condition: Array,
    ) -> Array:
        route_condition = self.local_context_norm(context_condition.astype(jnp.float32)).astype(jnp.float32)
        q_hint = trajectory_condition.astype(jnp.float32)
        local_base = (
            self.local_norm(h.astype(jnp.float32)).astype(jnp.float32)
            + route_condition
        ).astype(self.dtype)
        # Local mixing has fixed spatial routing, so it can safely consume the q hint.
        local_input = (local_base.astype(jnp.float32) + q_hint).astype(self.dtype)
        local = self.local_mlp(local_input).astype(jnp.float32)
        return (h.astype(jnp.float32) + self.local_scale * local).astype(self.dtype)


class BRCModel(nnx.Module):
    """Q-state recurrent solver for fixed-size grid reasoning tasks.

    The recurrent state stores centered log-q logits. Its softmax is the
    per-cell explicit answer distribution, which is read as a typed hypothesis
    hint; BRC does not learn a separate per-cell confidence state.
    """

    def __init__(
        self,
        config: ModelConfig,
        runtime: RuntimeConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if config.model_type != "brc":
            raise ValueError("BRCModel requires model_type='brc'")
        if config.grid_height * config.grid_width != config.seq_len:
            raise ValueError("grid_height * grid_width must equal seq_len")
        if config.task_type == "sudoku" and config.vocab_size != 9:
            raise ValueError("BRC Sudoku expects output vocab_size=9")
        brc = config.brc_config
        if brc.commit_steps < 1:
            raise ValueError("BRC commit_steps must be at least 1")
        if min(brc.refine_steps, brc.block_depth) < 1:
            raise ValueError("BRC refine_steps and block_depth must be at least 1")
        hidden_dim = int(brc.hidden_state_dim) if brc.hidden_state_dim > 0 else config.d_model
        if hidden_dim < 1:
            raise ValueError("BRC hidden_state_dim must be positive or 0 for d_model")
        if brc.num_heads < 1:
            raise ValueError("BRC num_heads must be at least 1")
        if hidden_dim % brc.num_heads != 0:
            raise ValueError("BRC hidden state dimension must be divisible by num_heads")
        if brc.mlp_ratio < 1:
            raise ValueError("BRC mlp_ratio must be at least 1")
        if brc.local_kernel < 1 or brc.local_kernel % 2 == 0:
            raise ValueError("BRC local_kernel must be a positive odd integer")
        if brc.position_encoding not in ("rope", "learned", "none"):
            raise ValueError("BRC position_encoding must be 'rope', 'learned', or 'none'")
        if brc.position_encoding == "rope" and (hidden_dim // brc.num_heads) % 4 != 0:
            raise ValueError("BRC axial RoPE head dimension must be divisible by 4")
        if brc.step_loss_schedule not in ("uniform", "linear", "quadratic"):
            raise ValueError("BRC step_loss_schedule must be 'uniform', 'linear', or 'quadratic'")
        if brc.update_rule not in ("energy", "velocity"):
            raise ValueError("BRC update_rule must be 'energy' or 'velocity'")
        if brc.descent_step_size <= 0.0:
            raise ValueError("BRC descent_step_size must be positive")
        if brc.descent_rms_clip <= 0.0:
            raise ValueError("BRC descent_rms_clip must be positive")
        if brc.fixed_point_label_smoothing < 0.0 or brc.fixed_point_label_smoothing >= 1.0:
            raise ValueError("BRC fixed_point_label_smoothing must be in [0, 1)")
        if min(
            brc.path_energy_weight,
            brc.fixed_point_update_weight,
            brc.wrong_attractor_rank_weight,
            brc.wrong_attractor_direction_weight,
            brc.wrong_attractor_nonzero_weight,
            brc.corrupted_recovery_weight,
        ) < 0.0:
            raise ValueError("BRC objective weights must be non-negative")
        if brc.wrong_attractor_rank_margin < 0.0:
            raise ValueError("BRC wrong_attractor_rank_margin must be non-negative")
        if brc.wrong_attractor_grad_floor < 0.0:
            raise ValueError("BRC wrong_attractor_grad_floor must be non-negative")
        if brc.early_stop_min_steps < 1:
            raise ValueError("BRC early_stop_min_steps must be at least 1")
        if brc.early_stop_patience < 1:
            raise ValueError("BRC early_stop_patience must be at least 1")
        if brc.early_stop_energy_q_delta_threshold < 0.0:
            raise ValueError("BRC early_stop_energy_q_delta_threshold must be non-negative")
        if brc.early_stop_flip_threshold < 0.0:
            raise ValueError("BRC early_stop_flip_threshold must be non-negative")
        if brc.early_stop_margin_threshold < 0.0:
            raise ValueError("BRC early_stop_margin_threshold must be non-negative")
        self.config = config
        self.runtime = runtime
        self.brc = brc
        self.commit_steps = int(brc.commit_steps)
        self.total_steps = self.commit_steps
        self.refine_steps = int(brc.refine_steps)
        self.trajectory_length = self.total_steps
        self.hidden_dim = hidden_dim
        self.dtype = compute_dtype(runtime.compute_dtype)
        self.embed_scale = math.sqrt(config.d_model)
        if config.task_type == "sudoku":
            self.input_vocab_size = int(config.input_vocab_size or 10)
            self.q_vocab_size = 9
            self.sudoku_blank_token_id = 0
        else:
            self.input_vocab_size = int(config.input_vocab_size or config.vocab_size)
            self.q_vocab_size = config.vocab_size
            self.sudoku_blank_token_id = 0
        self.output_logit_eps = 1e-9
        self.box_height, self.box_width = self._box_shape(config.grid_height, config.grid_width)

        rows = jnp.arange(config.seq_len, dtype=jnp.int32) // config.grid_width
        cols = jnp.arange(config.seq_len, dtype=jnp.int32) % config.grid_width
        boxes = (rows // self.box_height) * (config.grid_width // self.box_width) + (cols // self.box_width)
        box_indices = self._build_box_indices(config.grid_height, config.grid_width, self.box_height, self.box_width)
        self.row_ids = nnx.data(rows)
        self.col_ids = nnx.data(cols)
        self.box_ids = nnx.data(boxes)
        self.box_indices = nnx.data(box_indices)
        self.num_boxes = int(box_indices.shape[0])

        embed_init = trunc_normal_init(1.0 / self.embed_scale)
        self.puzzle_embed = nnx.Embed(
            self.input_vocab_size,
            config.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            embedding_init=embed_init,
            rngs=rngs,
        )
        self.context_embed = nnx.Embed(2, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
        self.trajectory_embed = nnx.Embed(
            max(10, self.q_vocab_size),
            config.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            embedding_init=embed_init,
            rngs=rngs,
        )
        if brc.position_encoding == "learned":
            self.row_embed = nnx.Embed(config.grid_height, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
            self.col_embed = nnx.Embed(config.grid_width, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
            self.box_embed = nnx.Embed(self.num_boxes, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        if brc.position_encoding == "rope":
            head_dim = self.hidden_dim // brc.num_heads
            axis_dim = head_dim // 2
            inv_freq = 1.0 / (brc.rope_theta ** (jnp.arange(0, axis_dim, 2, dtype=jnp.float32) / axis_dim))
            row_freqs = rows.astype(jnp.float32)[:, None] * inv_freq[None, :]
            col_freqs = cols.astype(jnp.float32)[:, None] * inv_freq[None, :]
            freqs = jnp.concatenate((row_freqs, col_freqs), axis=-1)
            rope = jnp.concatenate((freqs, freqs), axis=-1)
            self.rope_cos = nnx.data(jnp.cos(rope))
            self.rope_sin = nnx.data(jnp.sin(rope))
        else:
            self.rope_cos = None
            self.rope_sin = None

        self.context_to_hidden = nnx.Linear(
            config.d_model,
            self.hidden_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.trajectory_to_hidden = nnx.Linear(
            config.d_model,
            self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.energy_context_hidden_norm = unscaled_rms_norm(self.hidden_dim, brc.rms_norm_eps, self.dtype, rngs)
        self.energy_context_condition_norm = unscaled_rms_norm(self.hidden_dim, brc.rms_norm_eps, self.dtype, rngs)
        self.energy_context_output_norm = unscaled_rms_norm(self.hidden_dim, brc.rms_norm_eps, self.dtype, rngs)
        self.solver_blocks = nnx.List(
            [
                BRCSolverBlock(
                    config,
                    self.hidden_dim,
                    brc.num_heads,
                    brc.mlp_ratio,
                    self.dtype,
                    rngs=rngs,
                )
                for _ in range(brc.block_depth)
            ]
        )
        self.energy_state_to_hidden = nnx.Linear(
            self.q_vocab_size,
            self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.energy_norm = unscaled_rms_norm(self.hidden_dim, brc.rms_norm_eps, self.dtype, rngs)
        self.energy_hidden = nnx.Linear(
            self.hidden_dim,
            self.hidden_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.energy_head = nnx.Linear(
            self.hidden_dim,
            1,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.velocity_state_to_hidden = nnx.Linear(
            self.q_vocab_size,
            self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.velocity_norm = unscaled_rms_norm(self.hidden_dim, brc.rms_norm_eps, self.dtype, rngs)
        self.velocity_hidden = nnx.Linear(
            self.hidden_dim,
            self.hidden_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.velocity_head = nnx.Linear(
            self.hidden_dim,
            self.q_vocab_size,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
    @staticmethod
    def _box_shape(grid_height: int, grid_width: int) -> tuple[int, int]:
        box_height = int(math.sqrt(grid_height))
        while box_height > 1 and grid_height % box_height != 0:
            box_height -= 1
        box_width = int(math.sqrt(grid_width))
        while box_width > 1 and grid_width % box_width != 0:
            box_width -= 1
        return max(box_height, 1), max(box_width, 1)

    @staticmethod
    def _build_box_indices(grid_height: int, grid_width: int, box_height: int, box_width: int) -> Array:
        indices: list[list[int]] = []
        for box_row in range(grid_height // box_height):
            for box_col in range(grid_width // box_width):
                cells = []
                for row in range(box_row * box_height, (box_row + 1) * box_height):
                    for col in range(box_col * box_width, (box_col + 1) * box_width):
                        cells.append(row * grid_width + col)
                indices.append(cells)
        return jnp.asarray(indices, dtype=jnp.int32)

    def context_mask(self, tokens: Array) -> Array:
        if self.config.task_type == "sudoku":
            return tokens > self.sudoku_blank_token_id
        return tokens != 0

    def _center_q_logits(self, q_logits: Array) -> Array:
        q_logits = q_logits.astype(jnp.float32)
        return q_logits - jnp.mean(q_logits, axis=-1, keepdims=True)

    def _normalize_q(self, q_logits: Array) -> Array:
        """Return the explicit answer distribution from the stored log-q state."""
        return jax.nn.softmax(q_logits.astype(jnp.float32), axis=-1)

    def _uniform_q(self, tokens: Array) -> Array:
        # Zero centered logits represent the uniform answer distribution.
        return jnp.zeros(
            (*tokens.shape, self.q_vocab_size),
            dtype=jnp.float32,
        )

    def _q_to_output_logits(self, q_logits: Array, tokens: Array) -> Array:
        del tokens
        return self._center_q_logits(q_logits)

    def initial_q(self, tokens: Array) -> Array:
        return self._uniform_q(tokens)

    def initial_trajectory(self, tokens: Array, q_logits: Array | None = None) -> Array:
        if q_logits is None:
            q_logits = self.initial_q(tokens)
        return jnp.repeat(q_logits[:, None, :, :], self.trajectory_length, axis=1)

    def initial_trajectory_count(self, tokens: Array) -> Array:
        return jnp.zeros((tokens.shape[0],), dtype=jnp.int32)

    def _history_at(self, trajectory: Array, index: Array) -> Array:
        gather_index = jnp.broadcast_to(
            index[:, None, None, None],
            (trajectory.shape[0], 1, trajectory.shape[2], trajectory.shape[3]),
        )
        gathered = jnp.take_along_axis(
            trajectory,
            gather_index,
            axis=1,
        )
        return gathered[:, 0]

    def _append_trajectory(
        self,
        trajectory: Array,
        q_logits: Array,
        trajectory_count: Array,
    ) -> tuple[Array, Array]:
        q_logits = q_logits.astype(jnp.float32)
        next_count = jnp.minimum(trajectory_count + 1, self.trajectory_length)
        insert_index = jnp.minimum(trajectory_count, self.trajectory_length - 1)
        insert_mask = jax.nn.one_hot(insert_index, self.trajectory_length, dtype=bool)
        next_trajectory = jnp.where(insert_mask[:, :, None, None], q_logits[:, None, :, :], trajectory)
        return next_trajectory, next_count

    def _position_embeddings(self) -> Array:
        if self.brc.position_encoding != "learned":
            return jnp.zeros((self.config.seq_len, self.config.d_model), dtype=self.dtype)
        return (
            self.row_embed(self.row_ids)
            + self.col_embed(self.col_ids)
            + self.box_embed(self.box_ids)
        )

    def _trajectory_embedding(
        self,
        tokens: Array,
        trajectory: Array,
        trajectory_count: Array,
        step_index: Array,
    ) -> Array:
        del tokens, step_index
        time_index = jnp.arange(self.trajectory_length, dtype=jnp.int32)
        valid = time_index[None, :] < trajectory_count[:, None]
        valid_f = valid.astype(jnp.float32)[:, :, None, None]
        count = jnp.maximum(trajectory_count.astype(jnp.float32), 1.0)[:, None, None]
        current_index = jnp.maximum(trajectory_count - 1, 0)
        previous_index = jnp.maximum(trajectory_count - 2, 0)
        previous2_index = jnp.maximum(trajectory_count - 3, 0)
        has_current = (trajectory_count >= 1)[:, None, None]
        has_previous = (trajectory_count >= 2)[:, None, None]
        has_previous2 = (trajectory_count >= 3)[:, None, None]

        centered_trajectory = self._center_q_logits(trajectory)
        trajectory_q = jax.nn.softmax(centered_trajectory, axis=-1)
        uniform = jnp.full_like(trajectory_q, 1.0 / float(self.q_vocab_size))
        trajectory_direction = jnp.where(valid_f.astype(bool), trajectory_q - uniform, 0.0)
        mean_direction = jnp.sum(trajectory_direction, axis=1) / count

        q_logits = self._center_q_logits(self._history_at(trajectory, current_index))
        q_logits = jnp.where(has_current, q_logits, jnp.zeros_like(q_logits))
        q = self._normalize_q(q_logits)
        latest_direction = jnp.where(has_current, q - uniform[:, 0], jnp.zeros_like(q))
        direction = latest_direction + mean_direction

        previous = self._center_q_logits(self._history_at(trajectory, previous_index))
        previous2 = self._center_q_logits(self._history_at(trajectory, previous2_index))
        latest_delta = self._center_q_logits(q_logits - previous)
        latest_delta = jnp.where(has_previous, latest_delta, jnp.zeros_like(latest_delta))
        pair_valid = (time_index[None, 1:] < trajectory_count[:, None]).astype(jnp.float32)[:, :, None, None]
        pair_count = jnp.maximum((trajectory_count - 1).astype(jnp.float32), 1.0)[:, None, None]
        trajectory_delta = self._center_q_logits(centered_trajectory[:, 1:] - centered_trajectory[:, :-1])
        mean_delta = jnp.sum(trajectory_delta * pair_valid, axis=1) / pair_count
        history_delta = latest_delta + mean_delta

        latest_acceleration = self._center_q_logits(q_logits - 2.0 * previous + previous2)
        latest_acceleration = jnp.where(has_previous2, latest_acceleration, jnp.zeros_like(latest_acceleration))
        accel_valid = (time_index[None, 2:] < trajectory_count[:, None]).astype(jnp.float32)[:, :, None, None]
        accel_count = jnp.maximum((trajectory_count - 2).astype(jnp.float32), 1.0)[:, None, None]
        trajectory_acceleration = self._center_q_logits(
            centered_trajectory[:, 2:] - 2.0 * centered_trajectory[:, 1:-1] + centered_trajectory[:, :-2]
        )
        mean_acceleration = jnp.sum(trajectory_acceleration * accel_valid, axis=1) / accel_count
        acceleration = latest_acceleration + mean_acceleration
        embedding_table = maybe_cast(self.trajectory_embed.embedding[: self.q_vocab_size], self.dtype)
        direction_embedding = jnp.einsum(
            "bnd,dk->bnk",
            maybe_cast(direction, self.dtype),
            embedding_table,
            preferred_element_type=jnp.float32,
        )
        delta_embedding = jnp.einsum(
            "bnd,dk->bnk",
            maybe_cast(jnp.tanh(history_delta), self.dtype),
            embedding_table,
            preferred_element_type=jnp.float32,
        )
        acceleration_embedding = jnp.einsum(
            "bnd,dk->bnk",
            maybe_cast(jnp.tanh(acceleration), self.dtype),
            embedding_table,
            preferred_element_type=jnp.float32,
        )
        return direction_embedding + delta_embedding + acceleration_embedding

    def _typed_conditions(
        self,
        tokens: Array,
        trajectory: Array,
        trajectory_count: Array,
        base_embeddings: Array,
        step_index: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> tuple[Array, Array]:
        trajectory_embedding = self._trajectory_embedding(tokens, trajectory, trajectory_count, step_index)
        context_input = self.dropout(base_embeddings, deterministic=not train, rngs=dropout_key)
        q_input = self.dropout(trajectory_embedding, deterministic=not train, rngs=dropout_key)
        context_condition = self.context_to_hidden(maybe_cast(context_input, self.dtype)).astype(self.dtype)
        trajectory_condition = self.trajectory_to_hidden(maybe_cast(q_input, self.dtype)).astype(self.dtype)
        return context_condition, trajectory_condition

    def _q_to_token_logits(self, q: Array, tokens: Array, step_index: Array) -> Array:
        del step_index
        return self._q_to_output_logits(q, tokens)

    def _energy_descent_step(
        self,
        tokens: Array,
        q_logits: Array,
        energy_context: Array,
        step_index: Array,
    ) -> tuple[Array, dict[str, Array]]:
        del tokens, step_index
        current_logits = self._center_q_logits(q_logits)
        return self._energy_descent_from_context(current_logits, energy_context)

    def _energy_per_cell_from_context(
        self,
        candidate_q: Array,
        energy_context: Array,
    ) -> Array:
        uniform = jnp.full_like(candidate_q, 1.0 / float(self.q_vocab_size))
        trajectory_condition = self.energy_state_to_hidden(
            maybe_cast(candidate_q - uniform, self.dtype)
        ).astype(jnp.float32)
        energy_input = self.energy_norm(
            (energy_context.astype(jnp.float32) + trajectory_condition).astype(self.dtype)
        )
        energy_hidden = jax.nn.silu(
            self.energy_hidden(maybe_cast(energy_input, self.dtype)).astype(jnp.float32)
        )
        return self.energy_head(maybe_cast(energy_hidden, self.dtype)).astype(jnp.float32)[..., 0]

    def _energy_value_for_logits(
        self,
        q_logits: Array,
        energy_context: Array,
    ) -> Array:
        candidate_q = jax.nn.softmax(self._center_q_logits(q_logits), axis=-1)
        return self._energy_per_cell_from_context(candidate_q, energy_context)

    def _direct_velocity_step_from_context(
        self,
        current_logits: Array,
        energy_context: Array,
    ) -> tuple[Array, dict[str, Array]]:
        current_logits = self._center_q_logits(current_logits)
        current_q = jax.nn.softmax(current_logits, axis=-1)
        uniform = jnp.full_like(current_q, 1.0 / float(self.q_vocab_size))
        state_condition = self.velocity_state_to_hidden(
            maybe_cast(current_q - uniform, self.dtype)
        ).astype(jnp.float32)
        velocity_input = self.velocity_norm(
            (energy_context.astype(jnp.float32) + state_condition).astype(self.dtype)
        )
        velocity_hidden = jax.nn.silu(
            self.velocity_hidden(maybe_cast(velocity_input, self.dtype)).astype(jnp.float32)
        )
        raw_velocity = self.velocity_head(maybe_cast(velocity_hidden, self.dtype)).astype(jnp.float32)
        velocity = self._center_q_logits(raw_velocity)
        raw_logit_step = float(self.brc.descent_step_size) * velocity
        raw_step_rms = jnp.sqrt(jnp.mean(jnp.square(raw_logit_step), axis=-1, keepdims=True) + 1e-12)
        clip_scale = jnp.minimum(1.0, float(self.brc.descent_rms_clip) / raw_step_rms)
        logit_step = raw_logit_step * clip_scale
        next_logits = self._center_q_logits(current_logits + logit_step)
        next_q = jax.nn.softmax(next_logits, axis=-1)

        distribution_tv_delta = 0.5 * jnp.sum(jnp.abs(next_q - current_q), axis=-1, keepdims=True)
        velocity_rms = jnp.sqrt(jnp.mean(jnp.square(velocity), axis=-1, keepdims=True) + 1e-12)
        logit_step_rms = jnp.sqrt(jnp.mean(jnp.square(logit_step), axis=-1, keepdims=True) + 1e-12)
        path_energy = jnp.mean(jnp.square(logit_step), axis=-1, keepdims=True)
        descent_step_size = jnp.broadcast_to(
            jnp.asarray(float(self.brc.descent_step_size), dtype=jnp.float32),
            (*current_logits.shape[:-1], 1),
        )
        diagnostics = {
            "update_step_size": descent_step_size,
            "update_clip_scale": clip_scale,
            "update_rms": velocity_rms,
            "velocity_rms": velocity_rms,
            "velocity_clip_scale": clip_scale,
            "energy_update_rms": jnp.zeros_like(velocity_rms),
            "energy_value": jnp.zeros_like(velocity_rms),
            "energy_grad_rms": jnp.zeros_like(velocity_rms),
            "descent_step_size": descent_step_size,
            "descent_clip_scale": clip_scale,
            "descent_rms": velocity_rms,
            "logit_step_rms": logit_step_rms,
            "distribution_tv_delta": distribution_tv_delta,
            "path_energy": path_energy,
        }
        return next_logits, diagnostics

    def _energy_descent_from_context(
        self,
        current_logits: Array,
        energy_context: Array,
    ) -> tuple[Array, dict[str, Array]]:
        current_logits = self._center_q_logits(current_logits)
        current_q = jax.nn.softmax(current_logits, axis=-1)

        def energy_per_cell(candidate_q: Array) -> Array:
            return self._energy_per_cell_from_context(candidate_q, energy_context)

        def total_energy(candidate_q: Array) -> Array:
            return jnp.sum(energy_per_cell(candidate_q))

        energy_value = energy_per_cell(current_q)[..., None]
        energy_grad = self._center_q_logits(jax.grad(total_energy)(current_q))
        descent_direction = -energy_grad
        raw_logit_step = float(self.brc.descent_step_size) * descent_direction
        raw_step_rms = jnp.sqrt(jnp.mean(jnp.square(raw_logit_step), axis=-1, keepdims=True) + 1e-12)
        clip_scale = jnp.minimum(1.0, float(self.brc.descent_rms_clip) / raw_step_rms)
        logit_step = raw_logit_step * clip_scale
        descent_step_size = jnp.broadcast_to(
            jnp.asarray(float(self.brc.descent_step_size), dtype=jnp.float32),
            (*current_logits.shape[:-1], 1),
        )
        descent_clip_scale = clip_scale
        next_logits = current_logits + logit_step
        next_logits = self._center_q_logits(next_logits)
        next_q = jax.nn.softmax(next_logits, axis=-1)

        distribution_tv_delta = 0.5 * jnp.sum(jnp.abs(next_q - current_q), axis=-1, keepdims=True)
        energy_grad_rms = jnp.sqrt(jnp.mean(jnp.square(energy_grad), axis=-1, keepdims=True) + 1e-12)
        descent_rms = jnp.sqrt(jnp.mean(jnp.square(descent_direction), axis=-1, keepdims=True) + 1e-12)
        logit_step_rms = jnp.sqrt(jnp.mean(jnp.square(logit_step), axis=-1, keepdims=True) + 1e-12)
        path_energy = jnp.mean(jnp.square(logit_step), axis=-1, keepdims=True)
        diagnostics = {
            "update_step_size": descent_step_size,
            "update_clip_scale": descent_clip_scale,
            "update_rms": descent_rms,
            "velocity_rms": jnp.zeros_like(descent_rms),
            "velocity_clip_scale": jnp.ones_like(descent_clip_scale),
            "energy_update_rms": descent_rms,
            "energy_value": energy_value,
            "energy_grad_rms": energy_grad_rms,
            "descent_step_size": descent_step_size,
            "descent_clip_scale": descent_clip_scale,
            "descent_rms": descent_rms,
            "logit_step_rms": logit_step_rms,
            "distribution_tv_delta": distribution_tv_delta,
            "path_energy": path_energy,
        }
        return next_logits, diagnostics

    def _hidden_h_cycle(
        self,
        hidden_state: Array,
        context_condition: Array,
        trajectory_condition: Array,
    ) -> Array:
        hidden = hidden_state.astype(self.dtype)
        # Within an h-cycle, cheap local propagation runs for every refine step.
        # The expensive all-to-all attention is reserved for the cycle boundary,
        # immediately before q commit / early-stop check.
        for _ in range(self.refine_steps):
            for block in self.solver_blocks:
                hidden = block.local_think(hidden, context_condition, trajectory_condition)
        for block in self.solver_blocks:
            hidden = block.global_communicate(
                hidden,
                context_condition,
                trajectory_condition,
                self.rope_cos,
                self.rope_sin,
            )
        return hidden

    def _energy_context(
        self,
        hidden_state: Array,
        context_condition: Array,
        trajectory_condition: Array,
    ) -> Array:
        hidden = self.energy_context_hidden_norm(hidden_state.astype(jnp.float32)).astype(jnp.float32)
        context_condition = self.energy_context_condition_norm(
            context_condition.astype(jnp.float32)
        ).astype(jnp.float32)
        base = self.energy_context_output_norm(hidden + context_condition).astype(self.dtype)
        # This is the fixed context Phi for the black-box scalar energy.
        return (base.astype(jnp.float32) + trajectory_condition.astype(jnp.float32)).astype(self.dtype)

    def _commit_step(
        self,
        tokens: Array,
        q: Array,
        trajectory: Array,
        trajectory_count: Array,
        hidden_state: Array,
        base_embeddings: Array,
        step_index: Array,
        *,
        train: bool,
        dropout_key: Array | None,
        stop_hidden_between_steps: bool = True,
    ) -> tuple[Array, Array, dict[str, Array]]:
        update_diagnostics = {
            "update_step_size": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "update_clip_scale": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "update_rms": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "velocity_rms": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "velocity_clip_scale": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "energy_update_rms": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "descent_step_size": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "descent_clip_scale": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "energy_value": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "energy_grad_rms": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "descent_rms": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "logit_step_rms": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "distribution_tv_delta": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "path_energy": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
        }
        # q/history are stopped before context construction: attention builds Phi,
        # while the differentiable energy descent only flows through E(q; Phi).
        context_condition, trajectory_condition = self._typed_conditions(
            tokens,
            jax.lax.stop_gradient(trajectory),
            jax.lax.stop_gradient(trajectory_count),
            base_embeddings,
            step_index,
            train=train,
            dropout_key=dropout_key,
        )
        hidden_state = jax.lax.stop_gradient(hidden_state) if stop_hidden_between_steps else hidden_state
        hidden_state = self._hidden_h_cycle(
            hidden_state,
            context_condition,
            trajectory_condition,
        )
        energy_context = self._energy_context(
            hidden_state,
            context_condition,
            trajectory_condition,
        )
        del step_index
        if self.brc.update_rule == "velocity":
            q, update_diagnostics = self._direct_velocity_step_from_context(q, energy_context)
        else:
            q, update_diagnostics = self._energy_descent_from_context(q, energy_context)
        return (
            q,
            hidden_state,
            update_diagnostics,
        )

    def target_q_logits(self, targets: Array) -> Array:
        """Return centered logits for the smoothed fixed-point answer distribution."""
        smoothing = float(self.brc.fixed_point_label_smoothing)
        labels = jnp.clip(targets.astype(jnp.int32), 0, self.q_vocab_size - 1)
        target_distribution = (
            (1.0 - smoothing) * jax.nn.one_hot(labels, self.q_vocab_size, dtype=jnp.float32)
            + smoothing / float(self.q_vocab_size)
        )
        return self._center_q_logits(jnp.log(jnp.maximum(target_distribution, self.output_logit_eps)))

    def fixed_point_update_loss(
        self,
        tokens: Array,
        targets: Array,
        loss_mask: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> Array:
        """Penalize nonzero update at the smoothed target fixed point.

        This auxiliary fixed-point probe asks the configured q update to vanish
        when q is already the smoothed target distribution. It does not change
        the main q trajectory state.
        """
        target_logits = self.target_q_logits(targets)
        base_embeddings, _context = self.context_memory(tokens)
        target_trajectory = self.initial_trajectory(tokens, target_logits)
        target_trajectory_count = jnp.full((tokens.shape[0],), self.trajectory_length, dtype=jnp.int32)
        hidden = self.initial_hidden_state(
            tokens,
            target_logits,
            base_embeddings,
            train=train,
            dropout_key=dropout_key,
        )
        final_step = jnp.asarray(self.total_steps - 1, dtype=jnp.int32)
        _next_q, _next_hidden, diagnostics = self._commit_step(
            tokens,
            target_logits,
            target_trajectory,
            target_trajectory_count,
            hidden,
            base_embeddings,
            final_step,
            train=train,
            dropout_key=dropout_key,
            stop_hidden_between_steps=True,
        )
        mask = loss_mask.astype(jnp.float32)
        cell_energy = diagnostics["path_energy"][..., 0].astype(jnp.float32)
        return jnp.sum(cell_energy * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    def _energy_context_for_state(
        self,
        tokens: Array,
        trajectory: Array,
        trajectory_count: Array,
        hidden_state: Array,
        base_embeddings: Array,
        step_index: Array,
        *,
        train: bool,
        dropout_key: Array | None,
        stop_hidden_between_steps: bool = True,
    ) -> tuple[Array, Array]:
        context_condition, trajectory_condition = self._typed_conditions(
            tokens,
            jax.lax.stop_gradient(trajectory),
            jax.lax.stop_gradient(trajectory_count),
            base_embeddings,
            step_index,
            train=train,
            dropout_key=dropout_key,
        )
        hidden_state = jax.lax.stop_gradient(hidden_state) if stop_hidden_between_steps else hidden_state
        hidden_state = self._hidden_h_cycle(
            hidden_state,
            context_condition,
            trajectory_condition,
        )
        return self._energy_context(
            hidden_state,
            context_condition,
            trajectory_condition,
        ), hidden_state

    def _masked_per_example_mean(self, values: Array, mask: Array) -> Array:
        mask_f = mask.astype(jnp.float32)
        return jnp.sum(values.astype(jnp.float32) * mask_f, axis=-1) / jnp.maximum(jnp.sum(mask_f, axis=-1), 1.0)

    def _masked_direction_cosine(
        self,
        direction: Array,
        target_direction: Array,
        mask: Array,
    ) -> Array:
        mask_f = mask.astype(jnp.float32)[..., None]
        dot = jnp.sum(direction.astype(jnp.float32) * target_direction.astype(jnp.float32) * mask_f, axis=(1, 2))
        direction_norm = jnp.sqrt(jnp.sum(jnp.square(direction.astype(jnp.float32)) * mask_f, axis=(1, 2)) + 1e-12)
        target_norm = jnp.sqrt(jnp.sum(jnp.square(target_direction.astype(jnp.float32)) * mask_f, axis=(1, 2)) + 1e-12)
        return dot / (direction_norm * target_norm + 1e-12)

    def _energy_recovery_terms(
        self,
        tokens: Array,
        targets: Array,
        loss_mask: Array,
        q_logits: Array,
        trajectory: Array,
        trajectory_count: Array,
        hidden_state: Array,
        base_embeddings: Array,
        step_index: Array,
        *,
        train: bool,
        dropout_key: Array | None,
        wrong_only: bool,
    ) -> dict[str, Array]:
        energy_context, _hidden = self._energy_context_for_state(
            tokens,
            trajectory,
            trajectory_count,
            hidden_state,
            base_embeddings,
            step_index,
            train=train,
            dropout_key=dropout_key,
            stop_hidden_between_steps=True,
        )
        current_logits = self._center_q_logits(q_logits)
        if self.brc.update_rule == "velocity":
            next_logits, diagnostics = self._direct_velocity_step_from_context(current_logits, energy_context)
        else:
            next_logits, diagnostics = self._energy_descent_from_context(current_logits, energy_context)
        target_logits = self.target_q_logits(targets)
        current_energy_cell = diagnostics["energy_value"][..., 0].astype(jnp.float32)
        target_energy_cell = self._energy_value_for_logits(target_logits, energy_context).astype(jnp.float32)
        mask = loss_mask.astype(bool)
        example_mask = jnp.sum(mask.astype(jnp.float32), axis=-1) > 0
        if wrong_only:
            pred = jnp.argmax(self._normalize_q(current_logits), axis=-1)
            exact = jnp.all(jnp.where(mask, pred == targets, True), axis=-1)
            example_mask = example_mask & (~exact)
        example_weight = example_mask.astype(jnp.float32)
        normalizer = jnp.maximum(jnp.sum(example_weight), 1.0)
        current_energy = self._masked_per_example_mean(current_energy_cell, mask)
        target_energy = self._masked_per_example_mean(target_energy_cell, mask)
        if self.brc.update_rule == "energy":
            rank_loss = jnp.sum(
                jax.nn.relu(float(self.brc.wrong_attractor_rank_margin) + target_energy - current_energy)
                * example_weight
            ) / normalizer
            energy_gap = jnp.sum((current_energy - target_energy) * example_weight) / normalizer
        else:
            rank_loss = jnp.asarray(0.0, dtype=jnp.float32)
            energy_gap = jnp.asarray(0.0, dtype=jnp.float32)

        logit_step = self._center_q_logits(next_logits - current_logits)
        target_delta = self._center_q_logits(target_logits - current_logits)
        cosine = self._masked_direction_cosine(logit_step, target_delta, mask)
        direction_loss = jnp.sum((1.0 - cosine) * example_weight) / normalizer

        grad_rms = self._masked_per_example_mean(diagnostics["energy_grad_rms"][..., 0], mask)
        nonzero_loss = jnp.sum(
            jnp.square(jax.nn.relu(float(self.brc.wrong_attractor_grad_floor) - grad_rms))
            * example_weight
        ) / normalizer
        active_rate = jnp.mean(example_weight)
        return {
            "rank_loss": rank_loss.astype(jnp.float32),
            "direction_loss": direction_loss.astype(jnp.float32),
            "nonzero_loss": nonzero_loss.astype(jnp.float32),
            "active_rate": active_rate.astype(jnp.float32),
            "direction_cosine": (jnp.sum(cosine * example_weight) / normalizer).astype(jnp.float32),
            "energy_gap": energy_gap.astype(jnp.float32),
        }

    def attractor_recovery_losses(
        self,
        tokens: Array,
        targets: Array,
        loss_mask: Array,
        q_logits: Array,
        trajectory: Array,
        trajectory_count: Array,
        hidden_state: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> dict[str, Array]:
        """Auxiliary losses that remove wrong low-energy attractors.

        The carry state is treated as replay from the model's own trajectory.
        A synthetic high-confidence wrong state is added as corrupted recovery.
        """
        base_embeddings, _context = self.context_memory(tokens)
        step_index = jnp.minimum(trajectory_count, self.total_steps - 1)
        carry_terms = self._energy_recovery_terms(
            tokens,
            targets,
            loss_mask,
            q_logits,
            trajectory,
            trajectory_count,
            hidden_state,
            base_embeddings,
            step_index,
            train=train,
            dropout_key=dropout_key,
            wrong_only=True,
        )

        wrong_labels = jnp.mod(targets.astype(jnp.int32) + 1, self.q_vocab_size)
        smoothing = float(self.brc.fixed_point_label_smoothing)
        wrong_distribution = (
            (1.0 - smoothing) * jax.nn.one_hot(wrong_labels, self.q_vocab_size, dtype=jnp.float32)
            + smoothing / float(self.q_vocab_size)
        )
        wrong_logits = self._center_q_logits(jnp.log(jnp.maximum(wrong_distribution, self.output_logit_eps)))
        wrong_trajectory = self.initial_trajectory(tokens, wrong_logits)
        wrong_trajectory_count = jnp.full((tokens.shape[0],), self.trajectory_length, dtype=jnp.int32)
        wrong_hidden = self.initial_hidden_state(
            tokens,
            wrong_logits,
            base_embeddings,
            train=train,
            dropout_key=dropout_key,
        )
        corrupted_terms = self._energy_recovery_terms(
            tokens,
            targets,
            loss_mask,
            wrong_logits,
            wrong_trajectory,
            wrong_trajectory_count,
            wrong_hidden,
            base_embeddings,
            jnp.asarray(self.total_steps - 1, dtype=jnp.int32),
            train=train,
            dropout_key=dropout_key,
            wrong_only=False,
        )
        return {
            "wrong_attractor_rank_loss": carry_terms["rank_loss"],
            "wrong_attractor_direction_loss": carry_terms["direction_loss"],
            "wrong_attractor_nonzero_loss": carry_terms["nonzero_loss"],
            "wrong_attractor_active_rate": carry_terms["active_rate"],
            "wrong_attractor_direction_cosine": carry_terms["direction_cosine"],
            "wrong_attractor_energy_gap": carry_terms["energy_gap"],
            "corrupted_recovery_loss": corrupted_terms["direction_loss"],
            "corrupted_recovery_rank_loss": corrupted_terms["rank_loss"],
            "corrupted_recovery_direction_cosine": corrupted_terms["direction_cosine"],
            "corrupted_recovery_energy_gap": corrupted_terms["energy_gap"],
        }

    def _sudoku_constraint_ok(self, prediction_digits: Array, inputs: Array) -> Array:
        given_mask = inputs > self.sudoku_blank_token_id
        given_digits = jnp.clip(inputs - 1, 0, self.q_vocab_size - 1)
        givens_ok = jnp.all(jnp.where(given_mask, prediction_digits == given_digits, True), axis=-1)
        grid = prediction_digits.reshape((prediction_digits.shape[0], self.config.grid_height, self.config.grid_width))
        row_ok = jnp.all(jnp.all(jnp.sum(jax.nn.one_hot(grid, self.q_vocab_size), axis=2) == 1, axis=-1), axis=-1)
        col_ok = jnp.all(jnp.all(jnp.sum(jax.nn.one_hot(grid, self.q_vocab_size), axis=1) == 1, axis=-1), axis=-1)
        flat = prediction_digits
        box_values = jnp.take(flat, self.box_indices, axis=1)
        box_ok = jnp.all(jnp.all(jnp.sum(jax.nn.one_hot(box_values, self.q_vocab_size), axis=2) == 1, axis=-1), axis=-1)
        return givens_ok & row_ok & col_ok & box_ok

    def _early_stop(
        self,
        q_old_logits: Array,
        q_new_logits: Array,
        inputs: Array,
        new_steps: Array,
        update_diagnostics: dict[str, Array],
    ) -> tuple[Array, dict[str, Array]]:
        old_q = self._normalize_q(q_old_logits)
        new_q = self._normalize_q(q_new_logits)
        query_mask = (~self.context_mask(inputs)).astype(jnp.float32)
        query_normalizer = jnp.maximum(jnp.sum(query_mask, axis=-1), 1.0)
        energy_q_delta_cell = 0.5 * jnp.sum(jnp.abs(new_q - old_q), axis=-1)
        energy_q_delta = jnp.sum(energy_q_delta_cell * query_mask, axis=-1) / query_normalizer
        old_pred = jnp.argmax(old_q, axis=-1)
        new_pred = jnp.argmax(new_q, axis=-1)
        flip_rate = jnp.sum((old_pred != new_pred).astype(jnp.float32) * query_mask, axis=-1) / query_normalizer
        energy_grad_cell = update_diagnostics["energy_grad_rms"][..., 0].astype(jnp.float32)
        energy_grad_rms = jnp.sum(energy_grad_cell * query_mask, axis=-1) / query_normalizer
        top2 = jnp.sort(new_q, axis=-1)[..., -2:]
        margin = top2[..., 1] - top2[..., 0]
        margin_min = jnp.min(jnp.where(query_mask.astype(bool), margin, jnp.inf), axis=-1)
        has_query = jnp.sum(query_mask, axis=-1) > 0
        margin_min = jnp.where(has_query, margin_min, jnp.max(margin, axis=-1))
        if self.config.task_type == "sudoku":
            constraint_ok = self._sudoku_constraint_ok(new_pred, inputs)
        else:
            constraint_ok = jnp.ones((inputs.shape[0],), dtype=bool)
        if not bool(self.brc.early_stop_require_constraints):
            constraint_ok = jnp.ones_like(constraint_ok)
        stable = (
            bool(self.brc.early_stop_enabled)
            & (new_steps >= int(self.brc.early_stop_min_steps))
            & (energy_q_delta <= float(self.brc.early_stop_energy_q_delta_threshold))
            & (flip_rate <= float(self.brc.early_stop_flip_threshold))
            & (margin_min >= float(self.brc.early_stop_margin_threshold))
            & constraint_ok
        )
        diagnostics = {
            "early_stop_energy_grad_rms": jnp.mean(energy_grad_rms.astype(jnp.float32)),
            "early_stop_energy_q_delta": jnp.mean(energy_q_delta.astype(jnp.float32)),
            "early_stop_flip_rate": jnp.mean(flip_rate.astype(jnp.float32)),
            "early_stop_margin_min": jnp.mean(margin_min.astype(jnp.float32)),
            "early_stop_constraint_rate": jnp.mean(constraint_ok.astype(jnp.float32)),
        }
        return stable, diagnostics

    def initial_hidden_state(
        self,
        tokens: Array,
        q: Array,
        base_embeddings: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> Array:
        del q, base_embeddings, train, dropout_key
        return jnp.zeros((tokens.shape[0], self.config.seq_len, self.hidden_dim), dtype=self.dtype)

    def context_memory(
        self,
        tokens: Array,
    ) -> tuple[Array, Array]:
        context = self.context_mask(tokens)
        token_ids = jnp.clip(tokens, 0, self.input_vocab_size - 1)
        base_embeddings = (
            self.puzzle_embed(token_ids.astype(jnp.int32))
            + self.context_embed(context.astype(jnp.int32))
        )
        if self.brc.position_encoding == "learned":
            base_embeddings = base_embeddings + self._position_embeddings()[None, :, :]
        return base_embeddings, context

    def initial_carry(self, batch: dict[str, Array]) -> dict[str, Array]:
        batch_size = batch["inputs"].shape[0]
        return {
            "q": self._uniform_q(batch["inputs"]),
            "trajectory": self.initial_trajectory(batch["inputs"]),
            "trajectory_count": self.initial_trajectory_count(batch["inputs"]),
            "hidden": jnp.zeros(
                (batch_size, self.config.seq_len, self.hidden_dim),
                dtype=self.dtype,
            ),
            "steps": jnp.zeros((batch_size,), dtype=jnp.int32),
            "stable_steps": jnp.zeros((batch_size,), dtype=jnp.int32),
            "reset": jnp.ones((batch_size,), dtype=bool),
            "current_inputs": jnp.zeros_like(batch["inputs"]),
            "current_labels": jnp.zeros_like(batch["labels"]),
            "current_example_mask": jnp.zeros((batch_size,), dtype=jnp.float32),
        }

    def forward_carry_step(
        self,
        carry: dict[str, Array],
        batch: dict[str, Array],
        *,
        train: bool,
        dropout_key: Array | None = None,
    ) -> tuple[dict[str, Array], Array, dict[str, Array]]:
        if dropout_key is None:
            dropout_key = jax.random.key(0)
        reset = carry["reset"]
        reset_cells = reset[:, None]
        reset_state = reset[:, None, None]
        inputs = jnp.where(reset_cells, batch["inputs"], carry["current_inputs"])
        labels = jnp.where(reset_cells, batch["labels"], carry["current_labels"])
        batch_example_mask = batch.get(
            "example_mask",
            jnp.ones((inputs.shape[0],), dtype=jnp.float32),
        ).astype(jnp.float32)
        example_mask = jnp.where(reset, batch_example_mask, carry["current_example_mask"])
        steps = jnp.where(reset, 0, carry["steps"])

        base_embeddings, _context = self.context_memory(inputs)
        step_index = jnp.minimum(steps, self.total_steps - 1)
        reset_q = self.initial_q(inputs)
        q = jnp.where(reset_state, reset_q, carry["q"])
        reset_trajectory = self.initial_trajectory(inputs, reset_q)
        reset_trajectory_count = self.initial_trajectory_count(inputs)
        trajectory = jnp.where(reset[:, None, None, None], reset_trajectory, carry["trajectory"])
        trajectory_count = jnp.where(reset, reset_trajectory_count, carry["trajectory_count"])
        stable_steps = jnp.where(reset, 0, carry["stable_steps"])
        reset_hidden = self.initial_hidden_state(
            inputs,
            reset_q,
            base_embeddings,
            train=train,
            dropout_key=dropout_key,
        )
        hidden = jnp.where(reset_state, reset_hidden, carry["hidden"])
        next_q, next_hidden, update_diagnostics = self._commit_step(
            inputs,
            q,
            trajectory,
            trajectory_count,
            hidden,
            base_embeddings,
            step_index,
            train=train,
            dropout_key=dropout_key,
            stop_hidden_between_steps=True,
        )
        logits = self._q_to_token_logits(next_q, inputs, step_index)
        new_steps = steps + 1
        next_trajectory, next_trajectory_count = self._append_trajectory(
            trajectory,
            next_q,
            trajectory_count,
        )
        is_last_step = new_steps >= self.total_steps
        early_stop, early_stop_diagnostics = self._early_stop(
            q,
            next_q,
            inputs,
            new_steps,
            update_diagnostics,
        )
        stable_steps = jnp.where(early_stop, stable_steps + 1, 0)
        stable_reset = stable_steps >= int(self.brc.early_stop_patience)
        next_reset = is_last_step | stable_reset
        new_carry = {
            "q": jax.lax.stop_gradient(next_q),
            "trajectory": jax.lax.stop_gradient(next_trajectory),
            "trajectory_count": jax.lax.stop_gradient(next_trajectory_count),
            "hidden": jax.lax.stop_gradient(next_hidden),
            "steps": jax.lax.stop_gradient(new_steps),
            "stable_steps": jax.lax.stop_gradient(stable_steps),
            "reset": jax.lax.stop_gradient(next_reset),
            "current_inputs": inputs,
            "current_labels": labels,
            "current_example_mask": example_mask,
        }
        diagnostics = {
            "carry_step": jnp.mean(new_steps.astype(jnp.float32)),
            "max_step_reset_rate": jnp.mean(is_last_step.astype(jnp.float32)),
            "early_stop_rate": jnp.mean(stable_reset.astype(jnp.float32)),
            "stability_rate": jnp.mean(early_stop.astype(jnp.float32)),
            "stable_steps": jnp.mean(stable_steps.astype(jnp.float32)),
            "reset_rate": jnp.mean(reset.astype(jnp.float32)),
            "update_step_size": jnp.mean(update_diagnostics["update_step_size"].astype(jnp.float32)),
            "update_clip_scale": jnp.mean(update_diagnostics["update_clip_scale"].astype(jnp.float32)),
            "update_rms": jnp.mean(update_diagnostics["update_rms"].astype(jnp.float32)),
            "velocity_rms": jnp.mean(update_diagnostics["velocity_rms"].astype(jnp.float32)),
            "velocity_clip_scale": jnp.mean(update_diagnostics["velocity_clip_scale"].astype(jnp.float32)),
            "energy_update_rms": jnp.mean(update_diagnostics["energy_update_rms"].astype(jnp.float32)),
            "descent_step_size": jnp.mean(update_diagnostics["descent_step_size"].astype(jnp.float32)),
            "descent_clip_scale": jnp.mean(update_diagnostics["descent_clip_scale"].astype(jnp.float32)),
            "energy_value": jnp.mean(update_diagnostics["energy_value"].astype(jnp.float32)),
            "energy_grad_rms": jnp.mean(update_diagnostics["energy_grad_rms"].astype(jnp.float32)),
            "descent_rms": jnp.mean(update_diagnostics["descent_rms"].astype(jnp.float32)),
            "logit_step_rms": jnp.mean(update_diagnostics["logit_step_rms"].astype(jnp.float32)),
            "distribution_tv_delta": jnp.mean(update_diagnostics["distribution_tv_delta"].astype(jnp.float32)),
            "path_energy": jnp.mean(update_diagnostics["path_energy"].astype(jnp.float32)),
            **early_stop_diagnostics,
        }
        return new_carry, logits, diagnostics

    def run_commit_steps(
        self,
        tokens: Array,
        *,
        initial_q: Array | None = None,
        train: bool,
        dropout_key: Array | None = None,
        return_final_only: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        if initial_q is None:
            initial_q = self.initial_q(tokens)
        initial_trajectory = self.initial_trajectory(tokens, initial_q)
        initial_trajectory_count = self.initial_trajectory_count(tokens)
        base_embeddings, context = self.context_memory(tokens)
        query_mask = (~context).astype(jnp.float32)
        query_normalizer = jnp.maximum(jnp.sum(query_mask), 1.0)

        def scan_step(carry, scan_inputs):
            step_index, step_dropout_key = scan_inputs
            q, trajectory, trajectory_count, hidden_state = carry
            next_q, next_hidden, update_diagnostics = self._commit_step(
                tokens,
                q,
                trajectory,
                trajectory_count,
                hidden_state,
                base_embeddings,
                step_index,
                train=train,
                dropout_key=step_dropout_key,
                stop_hidden_between_steps=True,
            )
            next_trajectory, next_trajectory_count = self._append_trajectory(
                trajectory,
                next_q,
                trajectory_count,
            )
            next_carry = (next_q, next_trajectory, next_trajectory_count, next_hidden)
            if return_final_only:
                return next_carry, None
            logits = self._q_to_token_logits(next_q, tokens, step_index)
            confidence = jnp.max(self._normalize_q(next_q), axis=-1)
            q_top1_probability = jnp.sum(confidence * query_mask) / query_normalizer
            return next_carry, (
                logits,
                q_top1_probability,
                jnp.mean(update_diagnostics["update_step_size"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["update_clip_scale"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["update_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["velocity_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["velocity_clip_scale"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["energy_update_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["descent_step_size"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["descent_clip_scale"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["energy_value"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["energy_grad_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["descent_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["logit_step_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["distribution_tv_delta"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["path_energy"].astype(jnp.float32)),
            )

        step_indices = jnp.arange(self.total_steps, dtype=jnp.int32)
        if dropout_key is None:
            step_dropout_keys = jax.random.split(jax.random.key(0), self.total_steps)
        else:
            step_dropout_keys = jax.random.split(dropout_key, self.total_steps)
        initial_hidden = self.initial_hidden_state(
            tokens,
            initial_q,
            base_embeddings,
            train=train,
            dropout_key=step_dropout_keys[0],
        )
        initial_carry = (
            initial_q.astype(jnp.float32),
            initial_trajectory.astype(jnp.float32),
            initial_trajectory_count,
            initial_hidden,
        )
        final_carry, scan_outputs = jax.lax.scan(
            scan_step,
            initial_carry,
            (step_indices, step_dropout_keys),
        )
        q_final, _trajectory_final, _trajectory_count_final, _hidden_final = final_carry
        if return_final_only:
            final_step = jnp.asarray(self.total_steps - 1, dtype=jnp.int32)
            logits = self._q_to_token_logits(q_final, tokens, final_step)
            diagnostics = {
                "q_top1_probability": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "unroll_steps": jnp.asarray(self.total_steps, dtype=jnp.float32),
                "update_step_size": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "update_clip_scale": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "update_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "velocity_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "velocity_clip_scale": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "energy_update_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "descent_step_size": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "descent_clip_scale": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "energy_value": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "energy_grad_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "descent_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "logit_step_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "distribution_tv_delta": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "path_energy": jnp.zeros((self.total_steps,), dtype=jnp.float32),
            }
            return logits, diagnostics
        (
            step_logits,
            q_top1_probability,
            update_step_size,
            update_clip_scale,
            update_rms,
            velocity_rms,
            velocity_clip_scale,
            energy_update_rms,
            descent_step_size,
            descent_clip_scale,
            energy_value,
            energy_grad_rms,
            descent_rms,
            logit_step_rms,
            distribution_tv_delta,
            path_energy,
        ) = scan_outputs
        diagnostics = {
            "q_top1_probability": q_top1_probability,
            "unroll_steps": jnp.asarray(self.total_steps, dtype=jnp.float32),
            "update_step_size": update_step_size,
            "update_clip_scale": update_clip_scale,
            "update_rms": update_rms,
            "velocity_rms": velocity_rms,
            "velocity_clip_scale": velocity_clip_scale,
            "energy_update_rms": energy_update_rms,
            "descent_step_size": descent_step_size,
            "descent_clip_scale": descent_clip_scale,
            "energy_value": energy_value,
            "energy_grad_rms": energy_grad_rms,
            "descent_rms": descent_rms,
            "logit_step_rms": logit_step_rms,
            "distribution_tv_delta": distribution_tv_delta,
            "path_energy": path_energy,
        }
        return step_logits, diagnostics

    def forward_all_steps_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        initial_q: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        return self.run_commit_steps(
            tokens,
            initial_q=initial_q,
            train=train,
            dropout_key=dropout_key,
        )

    def forward_final_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        initial_q: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        return self.forward_all_steps_with_diagnostics(
            tokens,
            train=train,
            dropout_key=dropout_key,
            initial_q=initial_q,
        )

    def forward_all_steps(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
    ) -> Array:
        logits, _ = self.forward_all_steps_with_diagnostics(tokens, train=train, dropout_key=dropout_key)
        return logits

    def forward_final(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
    ) -> Array:
        logits, _ = self.forward_final_with_diagnostics(tokens, train=train, dropout_key=dropout_key)
        return logits[-1]

    def __call__(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
    ) -> Array:
        return self.forward_final(tokens, train=train, dropout_key=dropout_key)
