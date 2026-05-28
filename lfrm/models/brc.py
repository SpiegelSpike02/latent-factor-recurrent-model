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
        state_condition: Array,
        rope_cos: Array | None,
        rope_sin: Array | None,
    ) -> Array:
        h = self.global_communicate(
            h,
            context_condition,
            state_condition,
            rope_cos,
            rope_sin,
        )
        return self.local_think(h, context_condition, state_condition)

    def global_communicate(
        self,
        h: Array,
        context_condition: Array,
        state_condition: Array,
        rope_cos: Array | None,
        rope_sin: Array | None,
    ) -> Array:
        route_condition = self.attn_context_norm(context_condition.astype(jnp.float32)).astype(jnp.float32)
        state_hint = state_condition.astype(jnp.float32)
        # Attention routing is anchored by the hidden workspace and hard context.
        # The stopped z signal is only a value hint for update construction.
        route = (
            self.attn_norm(h.astype(jnp.float32)).astype(jnp.float32)
            + route_condition
        ).astype(self.dtype)
        value = (route.astype(jnp.float32) + state_hint).astype(self.dtype)
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
            name="BRC state-conditioned attention",
        )
        return (h.astype(jnp.float32) + self.attn_scale * attn.astype(jnp.float32)).astype(self.dtype)

    def local_think(
        self,
        h: Array,
        context_condition: Array,
        state_condition: Array,
    ) -> Array:
        route_condition = self.local_context_norm(context_condition.astype(jnp.float32)).astype(jnp.float32)
        state_hint = state_condition.astype(jnp.float32)
        local_base = (
            self.local_norm(h.astype(jnp.float32)).astype(jnp.float32)
            + route_condition
        ).astype(self.dtype)
        # Local mixing has fixed spatial routing, so it can safely consume the z-derived state hint.
        local_input = (local_base.astype(jnp.float32) + state_hint).astype(self.dtype)
        local = self.local_mlp(local_input).astype(jnp.float32)
        return (h.astype(jnp.float32) + self.local_scale * local).astype(self.dtype)


class BRCModel(nnx.Module):
    """Z-state recurrent solver for fixed-size grid reasoning tasks.

    The recurrent state stores centered answer logits ``z``. Its softmax is the
    per-cell explicit answer distribution view, which is read as a typed
    hypothesis hint; BRC does not learn a separate per-cell confidence state.
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
        if brc.update_step_size <= 0.0:
            raise ValueError("BRC update_step_size must be positive")
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
        if brc.wrong_attractor_update_floor < 0.0:
            raise ValueError("BRC wrong_attractor_update_floor must be non-negative")
        if brc.early_stop_min_steps < 1:
            raise ValueError("BRC early_stop_min_steps must be at least 1")
        if brc.early_stop_patience < 1:
            raise ValueError("BRC early_stop_patience must be at least 1")
        if brc.early_stop_distribution_delta_threshold < 0.0:
            raise ValueError("BRC early_stop_distribution_delta_threshold must be non-negative")
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
        self.state_embed = nnx.Embed(
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
        self.state_to_hidden = nnx.Linear(
            config.d_model,
            self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.state_scalar_to_hidden = nnx.Linear(
            3,
            self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.update_readout_norm = unscaled_rms_norm(self.hidden_dim, brc.rms_norm_eps, self.dtype, rngs)
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
        self.energy_head = nnx.Linear(
            self.hidden_dim,
            1,
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

    def _center_logits(self, z_logits: Array) -> Array:
        z_logits = z_logits.astype(jnp.float32)
        return z_logits - jnp.mean(z_logits, axis=-1, keepdims=True)

    def _distribution_from_logits(self, z_logits: Array) -> Array:
        """Return the explicit answer distribution view from centered logits."""
        return jax.nn.softmax(z_logits.astype(jnp.float32), axis=-1)

    def _logits_from_distribution(self, distribution: Array) -> Array:
        return self._center_logits(jnp.log(jnp.maximum(distribution, self.output_logit_eps)))

    def _zero_z(self, tokens: Array) -> Array:
        # Zero centered logits represent the uniform answer distribution.
        return jnp.zeros(
            (*tokens.shape, self.q_vocab_size),
            dtype=jnp.float32,
        )

    def _z_to_output_logits(self, z_logits: Array, tokens: Array) -> Array:
        del tokens
        if self.brc.update_rule == "energy":
            return self._distribution_from_logits(z_logits)
        return self._center_logits(z_logits)

    def initial_z(self, tokens: Array) -> Array:
        return self._zero_z(tokens)

    def _position_embeddings(self) -> Array:
        if self.brc.position_encoding != "learned":
            return jnp.zeros((self.config.seq_len, self.config.d_model), dtype=self.dtype)
        return (
            self.row_embed(self.row_ids)
            + self.col_embed(self.col_ids)
            + self.box_embed(self.box_ids)
        )

    def _state_features(
        self,
        tokens: Array,
        z_logits: Array,
        step_index: Array,
    ) -> tuple[Array, Array]:
        del tokens
        z_logits = self._center_logits(z_logits)
        q_view = self._distribution_from_logits(z_logits)
        uniform = jnp.full_like(q_view, 1.0 / float(self.q_vocab_size))
        logit_direction = jnp.tanh(z_logits / 4.0)
        current_direction = q_view - uniform

        entropy = -jnp.sum(q_view * jnp.log(jnp.maximum(q_view, 1e-8)), axis=-1, keepdims=True) / jnp.log(
            float(self.q_vocab_size)
        )
        top2 = jnp.sort(q_view, axis=-1)[..., -2:]
        margin = top2[..., 1:2] - top2[..., 0:1]
        step_progress = (step_index.astype(jnp.float32) + 1.0) / float(self.total_steps)
        step_progress = jnp.broadcast_to(step_progress, z_logits.shape[:1])
        progress = jnp.broadcast_to(step_progress[:, None, None], entropy.shape)
        scalar_features = jnp.concatenate(
            (
                entropy.astype(jnp.float32),
                margin.astype(jnp.float32),
                progress.astype(jnp.float32),
            ),
            axis=-1,
        )
        embedding_table = maybe_cast(self.state_embed.embedding[: self.q_vocab_size], self.dtype)
        logit_embedding = jnp.einsum(
            "bnd,dk->bnk",
            maybe_cast(logit_direction, self.dtype),
            embedding_table,
            preferred_element_type=jnp.float32,
        )
        direction_embedding = jnp.einsum(
            "bnd,dk->bnk",
            maybe_cast(current_direction, self.dtype),
            embedding_table,
            preferred_element_type=jnp.float32,
        )
        return logit_embedding + direction_embedding, scalar_features

    def _typed_conditions(
        self,
        tokens: Array,
        z_logits: Array,
        base_embeddings: Array,
        step_index: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> tuple[Array, Array]:
        state_embedding, state_scalar_features = self._state_features(
            tokens,
            z_logits,
            step_index,
        )
        context_input = self.dropout(base_embeddings, deterministic=not train, rngs=dropout_key)
        state_input = self.dropout(state_embedding, deterministic=not train, rngs=dropout_key)
        context_condition = self.context_to_hidden(maybe_cast(context_input, self.dtype)).astype(self.dtype)
        state_condition = self.state_to_hidden(maybe_cast(state_input, self.dtype)).astype(self.dtype)
        state_condition = state_condition + self.state_scalar_to_hidden(
            maybe_cast(state_scalar_features, self.dtype)
        ).astype(self.dtype)
        return context_condition, state_condition

    def _z_to_token_logits(self, z: Array, tokens: Array, step_index: Array) -> Array:
        del step_index
        return self._z_to_output_logits(z, tokens)

    def _energy_descent_step(
        self,
        tokens: Array,
        z_logits: Array,
        read_state: Array,
        step_index: Array,
    ) -> tuple[Array, dict[str, Array]]:
        del tokens, step_index
        current_logits = self._center_logits(z_logits)
        return self._energy_descent_from_read_state(current_logits, read_state)

    def _energy_per_cell_from_read_state(
        self,
        candidate_distribution: Array,
        read_state: Array,
    ) -> Array:
        uniform = jnp.full_like(candidate_distribution, 1.0 / float(self.q_vocab_size))
        state_condition = self.energy_state_to_hidden(
            maybe_cast(candidate_distribution - uniform, self.dtype)
        ).astype(jnp.float32)
        energy_input = self.energy_norm(
            (read_state.astype(jnp.float32) + state_condition).astype(self.dtype)
        )
        return self.energy_head(maybe_cast(energy_input, self.dtype)).astype(jnp.float32)[..., 0]

    def _energy_value_for_logits(
        self,
        z_logits: Array,
        read_state: Array,
    ) -> Array:
        candidate_distribution = jax.nn.softmax(self._center_logits(z_logits), axis=-1)
        return self._energy_per_cell_from_read_state(candidate_distribution, read_state)

    def _velocity_step_from_read_state(
        self,
        current_logits: Array,
        read_state: Array,
    ) -> tuple[Array, dict[str, Array]]:
        current_logits = self._center_logits(current_logits)
        current_distribution = jax.nn.softmax(current_logits, axis=-1)
        raw_velocity = self.velocity_head(maybe_cast(read_state, self.dtype)).astype(jnp.float32)
        velocity = self._center_logits(raw_velocity)
        raw_logit_step = float(self.brc.update_step_size) * velocity
        logit_step = raw_logit_step
        next_logits = self._center_logits(current_logits + logit_step)
        next_distribution = jax.nn.softmax(next_logits, axis=-1)

        distribution_tv_delta = 0.5 * jnp.sum(
            jnp.abs(next_distribution - current_distribution),
            axis=-1,
            keepdims=True,
        )
        velocity_rms = jnp.sqrt(jnp.mean(jnp.square(velocity), axis=-1, keepdims=True) + 1e-12)
        logit_step_rms = jnp.sqrt(jnp.mean(jnp.square(logit_step), axis=-1, keepdims=True) + 1e-12)
        path_energy = jnp.mean(jnp.square(logit_step), axis=-1, keepdims=True)
        update_step_size = jnp.broadcast_to(
            jnp.asarray(float(self.brc.update_step_size), dtype=jnp.float32),
            (*current_logits.shape[:-1], 1),
        )
        diagnostics = {
            "update_step_size": update_step_size,
            "update_rms": velocity_rms,
            "velocity_rms": velocity_rms,
            "energy_update_rms": jnp.zeros_like(velocity_rms),
            "energy_value": jnp.zeros_like(velocity_rms),
            "energy_grad_rms": jnp.zeros_like(velocity_rms),
            "logit_step_rms": logit_step_rms,
            "distribution_tv_delta": distribution_tv_delta,
            "path_energy": path_energy,
        }
        return next_logits, diagnostics

    def _energy_descent_from_read_state(
        self,
        current_logits: Array,
        read_state: Array,
    ) -> tuple[Array, dict[str, Array]]:
        current_logits = self._center_logits(current_logits)
        current_distribution = jax.nn.softmax(current_logits, axis=-1)

        def energy_per_cell(candidate_distribution: Array) -> Array:
            return self._energy_per_cell_from_read_state(candidate_distribution, read_state)

        def total_energy(candidate_distribution: Array) -> Array:
            return jnp.sum(energy_per_cell(candidate_distribution))

        energy_value = energy_per_cell(current_distribution)[..., None]
        energy_grad = self._center_logits(jax.grad(total_energy)(current_distribution))
        descent_direction = -energy_grad
        logit_step = float(self.brc.update_step_size) * descent_direction
        update_step_size = jnp.broadcast_to(
            jnp.asarray(float(self.brc.update_step_size), dtype=jnp.float32),
            (*current_logits.shape[:-1], 1),
        )
        next_logits = self._center_logits(current_logits + logit_step)
        next_distribution = jax.nn.softmax(next_logits, axis=-1)

        distribution_tv_delta = 0.5 * jnp.sum(
            jnp.abs(next_distribution - current_distribution),
            axis=-1,
            keepdims=True,
        )
        energy_grad_rms = jnp.sqrt(jnp.mean(jnp.square(energy_grad), axis=-1, keepdims=True) + 1e-12)
        update_rms = jnp.sqrt(jnp.mean(jnp.square(descent_direction), axis=-1, keepdims=True) + 1e-12)
        logit_step_rms = jnp.sqrt(jnp.mean(jnp.square(logit_step), axis=-1, keepdims=True) + 1e-12)
        path_energy = jnp.mean(jnp.square(next_distribution - current_distribution), axis=-1, keepdims=True)
        diagnostics = {
            "update_step_size": update_step_size,
            "update_rms": update_rms,
            "velocity_rms": jnp.zeros_like(update_rms),
            "energy_update_rms": update_rms,
            "energy_value": energy_value,
            "energy_grad_rms": energy_grad_rms,
            "logit_step_rms": logit_step_rms,
            "distribution_tv_delta": distribution_tv_delta,
            "path_energy": path_energy,
        }
        return next_logits, diagnostics

    def _hidden_h_cycle(
        self,
        hidden_state: Array,
        context_condition: Array,
        state_condition: Array,
    ) -> Array:
        hidden = hidden_state.astype(self.dtype)
        # Within an h-cycle, cheap local propagation runs for every refine step.
        # The expensive all-to-all attention is reserved for the cycle boundary,
        # immediately before z update / early-stop check.
        for _ in range(self.refine_steps):
            for block in self.solver_blocks:
                hidden = block.local_think(hidden, context_condition, state_condition)
        for block in self.solver_blocks:
            hidden = block.global_communicate(
                hidden,
                context_condition,
                state_condition,
                self.rope_cos,
                self.rope_sin,
            )
        return hidden

    def _update_read_state(
        self,
        hidden_state: Array,
    ) -> Array:
        return self.update_readout_norm(hidden_state.astype(jnp.float32)).astype(self.dtype)

    def _commit_step(
        self,
        tokens: Array,
        z: Array,
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
            "update_rms": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "velocity_rms": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "energy_update_rms": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "energy_value": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "energy_grad_rms": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "logit_step_rms": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "distribution_tv_delta": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "path_energy": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
        }
        # The explicit z state is stopped before context construction; the
        # configured update rule then acts on logits using the refined H.
        context_condition, state_condition = self._typed_conditions(
            tokens,
            jax.lax.stop_gradient(z),
            base_embeddings,
            step_index,
            train=train,
            dropout_key=dropout_key,
        )
        hidden_state = jax.lax.stop_gradient(hidden_state) if stop_hidden_between_steps else hidden_state
        hidden_state = self._hidden_h_cycle(
            hidden_state,
            context_condition,
            state_condition,
        )
        read_state = self._update_read_state(hidden_state)
        del step_index
        if self.brc.update_rule == "velocity":
            z, update_diagnostics = self._velocity_step_from_read_state(z, read_state)
        else:
            z, update_diagnostics = self._energy_descent_from_read_state(z, read_state)
        return (
            z,
            hidden_state,
            update_diagnostics,
        )

    def target_z_logits(self, targets: Array) -> Array:
        """Return centered logits for the smoothed fixed-point answer distribution."""
        smoothing = float(self.brc.fixed_point_label_smoothing)
        labels = jnp.clip(targets.astype(jnp.int32), 0, self.q_vocab_size - 1)
        target_distribution = (
            (1.0 - smoothing) * jax.nn.one_hot(labels, self.q_vocab_size, dtype=jnp.float32)
            + smoothing / float(self.q_vocab_size)
        )
        return self._center_logits(jnp.log(jnp.maximum(target_distribution, self.output_logit_eps)))

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

        This auxiliary fixed-point probe asks the configured z update to vanish
        when z already represents the smoothed target distribution. It does not
        change the main z dynamics state.
        """
        target_logits = self.target_z_logits(targets)
        base_embeddings, _context = self.context_memory(tokens)
        hidden = self.initial_hidden_state(
            tokens,
            target_logits,
            base_embeddings,
            train=train,
            dropout_key=dropout_key,
        )
        final_step = jnp.asarray(self.total_steps - 1, dtype=jnp.int32)
        _next_z, _next_hidden, diagnostics = self._commit_step(
            tokens,
            target_logits,
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

    def _update_read_state_for_state(
        self,
        tokens: Array,
        z_logits: Array,
        hidden_state: Array,
        base_embeddings: Array,
        step_index: Array,
        *,
        train: bool,
        dropout_key: Array | None,
        stop_hidden_between_steps: bool = True,
    ) -> tuple[Array, Array]:
        context_condition, state_condition = self._typed_conditions(
            tokens,
            jax.lax.stop_gradient(z_logits),
            base_embeddings,
            step_index,
            train=train,
            dropout_key=dropout_key,
        )
        hidden_state = jax.lax.stop_gradient(hidden_state) if stop_hidden_between_steps else hidden_state
        hidden_state = self._hidden_h_cycle(
            hidden_state,
            context_condition,
            state_condition,
        )
        return self._update_read_state(hidden_state), hidden_state

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

    def _attractor_recovery_terms(
        self,
        tokens: Array,
        targets: Array,
        loss_mask: Array,
        z_logits: Array,
        hidden_state: Array,
        base_embeddings: Array,
        step_index: Array,
        *,
        train: bool,
        dropout_key: Array | None,
        wrong_only: bool,
    ) -> dict[str, Array]:
        read_state, _hidden = self._update_read_state_for_state(
            tokens,
            z_logits,
            hidden_state,
            base_embeddings,
            step_index,
            train=train,
            dropout_key=dropout_key,
            stop_hidden_between_steps=True,
        )
        current_logits = self._center_logits(z_logits)
        if self.brc.update_rule == "velocity":
            next_logits, diagnostics = self._velocity_step_from_read_state(current_logits, read_state)
        else:
            next_logits, diagnostics = self._energy_descent_from_read_state(current_logits, read_state)
        target_logits = self.target_z_logits(targets)
        current_energy_cell = diagnostics["energy_value"][..., 0].astype(jnp.float32)
        target_energy_cell = self._energy_value_for_logits(target_logits, read_state).astype(jnp.float32)
        mask = loss_mask.astype(bool)
        example_mask = jnp.sum(mask.astype(jnp.float32), axis=-1) > 0
        if wrong_only:
            pred = jnp.argmax(self._distribution_from_logits(current_logits), axis=-1)
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

        if self.brc.update_rule == "energy":
            current_distribution = self._distribution_from_logits(current_logits)
            next_distribution = self._distribution_from_logits(next_logits)
            target_distribution = self._distribution_from_logits(target_logits)
            update_direction = next_distribution - current_distribution
            target_direction = target_distribution - current_distribution
        else:
            update_direction = self._center_logits(next_logits - current_logits)
            target_direction = self._center_logits(target_logits - current_logits)
        cosine = self._masked_direction_cosine(update_direction, target_direction, mask)
        direction_loss = jnp.sum((1.0 - cosine) * example_weight) / normalizer

        update_rms = self._masked_per_example_mean(diagnostics["update_rms"][..., 0], mask)
        nonzero_loss = jnp.sum(
            jnp.square(jax.nn.relu(float(self.brc.wrong_attractor_update_floor) - update_rms))
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
        z_logits: Array,
        hidden_state: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> dict[str, Array]:
        """Auxiliary losses that remove wrong low-energy attractors.

        The carry z state is treated as replay from the model's own dynamics.
        A synthetic high-confidence wrong state is added as corrupted recovery.
        """
        base_embeddings, _context = self.context_memory(tokens)
        step_index = jnp.asarray(self.total_steps - 1, dtype=jnp.int32)
        carry_terms = self._attractor_recovery_terms(
            tokens,
            targets,
            loss_mask,
            z_logits,
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
        wrong_logits = self._center_logits(jnp.log(jnp.maximum(wrong_distribution, self.output_logit_eps)))
        wrong_hidden = self.initial_hidden_state(
            tokens,
            wrong_logits,
            base_embeddings,
            train=train,
            dropout_key=dropout_key,
        )
        corrupted_terms = self._attractor_recovery_terms(
            tokens,
            targets,
            loss_mask,
            wrong_logits,
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
        old_z: Array,
        new_z: Array,
        inputs: Array,
        new_steps: Array,
        update_diagnostics: dict[str, Array],
    ) -> tuple[Array, dict[str, Array]]:
        old_distribution = self._distribution_from_logits(old_z)
        new_distribution = self._distribution_from_logits(new_z)
        query_mask = (~self.context_mask(inputs)).astype(jnp.float32)
        query_normalizer = jnp.maximum(jnp.sum(query_mask, axis=-1), 1.0)
        distribution_delta_cell = 0.5 * jnp.sum(jnp.abs(new_distribution - old_distribution), axis=-1)
        distribution_delta = jnp.sum(distribution_delta_cell * query_mask, axis=-1) / query_normalizer
        old_pred = jnp.argmax(old_distribution, axis=-1)
        new_pred = jnp.argmax(new_distribution, axis=-1)
        flip_rate = jnp.sum((old_pred != new_pred).astype(jnp.float32) * query_mask, axis=-1) / query_normalizer
        update_rms_cell = update_diagnostics["update_rms"][..., 0].astype(jnp.float32)
        update_rms = jnp.sum(update_rms_cell * query_mask, axis=-1) / query_normalizer
        top2 = jnp.sort(new_distribution, axis=-1)[..., -2:]
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
            & (distribution_delta <= float(self.brc.early_stop_distribution_delta_threshold))
            & (flip_rate <= float(self.brc.early_stop_flip_threshold))
            & (margin_min >= float(self.brc.early_stop_margin_threshold))
            & constraint_ok
        )
        diagnostics = {
            "early_stop_update_rms": jnp.mean(update_rms.astype(jnp.float32)),
            "early_stop_distribution_delta": jnp.mean(distribution_delta.astype(jnp.float32)),
            "early_stop_flip_rate": jnp.mean(flip_rate.astype(jnp.float32)),
            "early_stop_margin_min": jnp.mean(margin_min.astype(jnp.float32)),
            "early_stop_constraint_rate": jnp.mean(constraint_ok.astype(jnp.float32)),
        }
        return stable, diagnostics

    def initial_hidden_state(
        self,
        tokens: Array,
        z: Array,
        base_embeddings: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> Array:
        del z, base_embeddings, train, dropout_key
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
            "z": self._zero_z(batch["inputs"]),
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
        reset_z = self.initial_z(inputs)
        z = jnp.where(reset_state, reset_z, carry["z"])
        stable_steps = jnp.where(reset, 0, carry["stable_steps"])
        reset_hidden = self.initial_hidden_state(
            inputs,
            reset_z,
            base_embeddings,
            train=train,
            dropout_key=dropout_key,
        )
        hidden = jnp.where(reset_state, reset_hidden, carry["hidden"])
        next_z, next_hidden, update_diagnostics = self._commit_step(
            inputs,
            z,
            hidden,
            base_embeddings,
            step_index,
            train=train,
            dropout_key=dropout_key,
            stop_hidden_between_steps=True,
        )
        logits = self._z_to_token_logits(next_z, inputs, step_index)
        new_steps = steps + 1
        is_last_step = new_steps >= self.total_steps
        early_stop, early_stop_diagnostics = self._early_stop(
            z,
            next_z,
            inputs,
            new_steps,
            update_diagnostics,
        )
        stable_steps = jnp.where(early_stop, stable_steps + 1, 0)
        stable_reset = stable_steps >= int(self.brc.early_stop_patience)
        next_reset = is_last_step | stable_reset
        new_carry = {
            "z": jax.lax.stop_gradient(next_z),
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
            "update_rms": jnp.mean(update_diagnostics["update_rms"].astype(jnp.float32)),
            "velocity_rms": jnp.mean(update_diagnostics["velocity_rms"].astype(jnp.float32)),
            "energy_update_rms": jnp.mean(update_diagnostics["energy_update_rms"].astype(jnp.float32)),
            "energy_value": jnp.mean(update_diagnostics["energy_value"].astype(jnp.float32)),
            "energy_grad_rms": jnp.mean(update_diagnostics["energy_grad_rms"].astype(jnp.float32)),
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
        initial_z: Array | None = None,
        train: bool,
        dropout_key: Array | None = None,
        return_final_only: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        if initial_z is None:
            initial_z = self.initial_z(tokens)
        base_embeddings, context = self.context_memory(tokens)
        query_mask = (~context).astype(jnp.float32)
        query_normalizer = jnp.maximum(jnp.sum(query_mask), 1.0)

        def scan_step(carry, scan_inputs):
            step_index, step_dropout_key = scan_inputs
            z, hidden_state = carry
            next_z, next_hidden, update_diagnostics = self._commit_step(
                tokens,
                z,
                hidden_state,
                base_embeddings,
                step_index,
                train=train,
                dropout_key=step_dropout_key,
                stop_hidden_between_steps=True,
            )
            next_carry = (next_z, next_hidden)
            if return_final_only:
                return next_carry, None
            logits = self._z_to_token_logits(next_z, tokens, step_index)
            confidence = jnp.max(self._distribution_from_logits(next_z), axis=-1)
            q_top1_probability = jnp.sum(confidence * query_mask) / query_normalizer
            return next_carry, (
                logits,
                q_top1_probability,
                jnp.mean(update_diagnostics["update_step_size"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["update_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["velocity_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["energy_update_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["energy_value"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["energy_grad_rms"].astype(jnp.float32)),
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
            initial_z,
            base_embeddings,
            train=train,
            dropout_key=step_dropout_keys[0],
        )
        initial_carry = (
            initial_z.astype(jnp.float32),
            initial_hidden,
        )
        final_carry, scan_outputs = jax.lax.scan(
            scan_step,
            initial_carry,
            (step_indices, step_dropout_keys),
        )
        z_final, _hidden_final = final_carry
        if return_final_only:
            final_step = jnp.asarray(self.total_steps - 1, dtype=jnp.int32)
            logits = self._z_to_token_logits(z_final, tokens, final_step)
            diagnostics = {
                "q_top1_probability": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "unroll_steps": jnp.asarray(self.total_steps, dtype=jnp.float32),
                "update_step_size": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "update_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "velocity_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "energy_update_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "energy_value": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "energy_grad_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "logit_step_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "distribution_tv_delta": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "path_energy": jnp.zeros((self.total_steps,), dtype=jnp.float32),
            }
            return logits, diagnostics
        (
            step_logits,
            q_top1_probability,
            update_step_size,
            update_rms,
            velocity_rms,
            energy_update_rms,
            energy_value,
            energy_grad_rms,
            logit_step_rms,
            distribution_tv_delta,
            path_energy,
        ) = scan_outputs
        diagnostics = {
            "q_top1_probability": q_top1_probability,
            "unroll_steps": jnp.asarray(self.total_steps, dtype=jnp.float32),
            "update_step_size": update_step_size,
            "update_rms": update_rms,
            "velocity_rms": velocity_rms,
            "energy_update_rms": energy_update_rms,
            "energy_value": energy_value,
            "energy_grad_rms": energy_grad_rms,
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
        initial_z: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        return self.run_commit_steps(
            tokens,
            initial_z=initial_z,
            train=train,
            dropout_key=dropout_key,
        )

    def forward_final_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        initial_z: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        return self.forward_all_steps_with_diagnostics(
            tokens,
            train=train,
            dropout_key=dropout_key,
            initial_z=initial_z,
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
