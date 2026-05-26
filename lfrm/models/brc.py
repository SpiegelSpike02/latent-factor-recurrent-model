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
        q_condition: Array,
        rope_cos: Array | None,
        rope_sin: Array | None,
    ) -> Array:
        h = self.global_communicate(
            h,
            context_condition,
            q_condition,
            rope_cos,
            rope_sin,
        )
        return self.local_think(h, context_condition, q_condition)

    def global_communicate(
        self,
        h: Array,
        context_condition: Array,
        q_condition: Array,
        rope_cos: Array | None,
        rope_sin: Array | None,
    ) -> Array:
        route_condition = self.attn_context_norm(context_condition.astype(jnp.float32)).astype(jnp.float32)
        q_hint = q_condition.astype(jnp.float32)
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
            name="BRC self attention",
        )
        return (h.astype(jnp.float32) + self.attn_scale * attn.astype(jnp.float32)).astype(self.dtype)

    def local_think(
        self,
        h: Array,
        context_condition: Array,
        q_condition: Array,
    ) -> Array:
        route_condition = self.local_context_norm(context_condition.astype(jnp.float32)).astype(jnp.float32)
        q_hint = q_condition.astype(jnp.float32)
        local_base = (
            self.local_norm(h.astype(jnp.float32)).astype(jnp.float32)
            + route_condition
        ).astype(self.dtype)
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
        if brc.q_window < 1:
            raise ValueError("BRC q_window must be at least 1")
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
        if brc.halt_min_steps < 1:
            raise ValueError("BRC halt_min_steps must be at least 1")

        self.config = config
        self.runtime = runtime
        self.brc = brc
        self.commit_steps = int(brc.commit_steps)
        self.total_steps = self.commit_steps
        self.refine_steps = int(brc.refine_steps)
        self.q_window = int(brc.q_window)
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
        self.q_embed = nnx.Embed(
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
        self.q_to_hidden = nnx.Linear(
            config.d_model,
            self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.readout_hidden_norm = unscaled_rms_norm(self.hidden_dim, brc.rms_norm_eps, self.dtype, rngs)
        self.readout_condition_norm = unscaled_rms_norm(self.hidden_dim, brc.rms_norm_eps, self.dtype, rngs)
        self.readout_output_norm = unscaled_rms_norm(self.hidden_dim, brc.rms_norm_eps, self.dtype, rngs)
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
        self.q_proposal_head = nnx.Linear(
            self.hidden_dim,
            self.q_vocab_size,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.q_update_head = nnx.Linear(
            self.hidden_dim,
            1,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.halt_head = nnx.Linear(
            self.hidden_dim,
            1,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.constant(-5.0),
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

    def initial_q_history(self, tokens: Array, q_logits: Array | None = None) -> Array:
        if q_logits is None:
            q_logits = self.initial_q(tokens)
        return jnp.repeat(q_logits[:, None, :, :], self.q_window, axis=1)

    def initial_q_history_count(self, tokens: Array) -> Array:
        return jnp.zeros((tokens.shape[0],), dtype=jnp.int32)

    def _history_at(self, q_history: Array, index: Array) -> Array:
        gather_index = jnp.broadcast_to(
            index[:, None, None, None],
            (q_history.shape[0], 1, q_history.shape[2], q_history.shape[3]),
        )
        gathered = jnp.take_along_axis(
            q_history,
            gather_index,
            axis=1,
        )
        return gathered[:, 0]

    def _append_q_history(
        self,
        q_history: Array,
        q_logits: Array,
        q_history_count: Array,
    ) -> tuple[Array, Array]:
        q_logits = q_logits.astype(jnp.float32)
        next_count = jnp.minimum(q_history_count + 1, self.q_window)
        if self.q_window == 1:
            return q_logits[:, None, :, :], next_count
        is_full = q_history_count >= self.q_window
        shifted = jnp.concatenate((q_history[:, 1:], q_logits[:, None, :, :]), axis=1)
        insert_index = jnp.minimum(q_history_count, self.q_window - 1)
        insert_mask = jax.nn.one_hot(insert_index, self.q_window, dtype=bool)
        inserted = jnp.where(insert_mask[:, :, None, None], q_logits[:, None, :, :], q_history)
        next_history = jnp.where(is_full[:, None, None, None], shifted, inserted)
        return next_history, next_count

    def _position_embeddings(self) -> Array:
        if self.brc.position_encoding != "learned":
            return jnp.zeros((self.config.seq_len, self.config.d_model), dtype=self.dtype)
        return (
            self.row_embed(self.row_ids)
            + self.col_embed(self.col_ids)
            + self.box_embed(self.box_ids)
        )

    def _q_embedding(
        self,
        tokens: Array,
        q_history: Array,
        q_history_count: Array,
        step_index: Array,
    ) -> Array:
        del tokens, step_index
        current_index = jnp.maximum(q_history_count - 1, 0)
        previous_index = jnp.maximum(q_history_count - 2, 0)
        previous2_index = jnp.maximum(q_history_count - 3, 0)
        has_current = (q_history_count >= 1)[:, None, None]
        has_previous = (q_history_count >= 2)[:, None, None]
        has_previous2 = (q_history_count >= 3)[:, None, None]

        q_logits = self._center_q_logits(self._history_at(q_history, current_index))
        q_logits = jnp.where(has_current, q_logits, jnp.zeros_like(q_logits))
        q = self._normalize_q(q_logits)
        uniform = jnp.full_like(q, 1.0 / float(self.q_vocab_size))
        direction = jnp.where(has_current, q - uniform, jnp.zeros_like(q))
        if self.q_window >= 2:
            previous = self._center_q_logits(self._history_at(q_history, previous_index))
            velocity = self._center_q_logits(q_logits - previous)
            velocity = jnp.where(has_previous, velocity, jnp.zeros_like(velocity))
        else:
            velocity = jnp.zeros_like(q_logits)
        if self.q_window >= 3:
            previous = self._center_q_logits(self._history_at(q_history, previous_index))
            previous2 = self._center_q_logits(self._history_at(q_history, previous2_index))
            acceleration = self._center_q_logits(q_logits - 2.0 * previous + previous2)
            acceleration = jnp.where(has_previous2, acceleration, jnp.zeros_like(acceleration))
        else:
            acceleration = jnp.zeros_like(q_logits)
        embedding_table = maybe_cast(self.q_embed.embedding[: self.q_vocab_size], self.dtype)
        direction_embedding = jnp.einsum(
            "bnd,dk->bnk",
            maybe_cast(direction, self.dtype),
            embedding_table,
            preferred_element_type=jnp.float32,
        )
        velocity_embedding = jnp.einsum(
            "bnd,dk->bnk",
            maybe_cast(jnp.tanh(velocity), self.dtype),
            embedding_table,
            preferred_element_type=jnp.float32,
        )
        acceleration_embedding = jnp.einsum(
            "bnd,dk->bnk",
            maybe_cast(jnp.tanh(acceleration), self.dtype),
            embedding_table,
            preferred_element_type=jnp.float32,
        )
        return direction_embedding + velocity_embedding + acceleration_embedding

    def _typed_conditions(
        self,
        tokens: Array,
        q_history: Array,
        q_history_count: Array,
        base_embeddings: Array,
        step_index: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> tuple[Array, Array]:
        q_embedding = self._q_embedding(tokens, q_history, q_history_count, step_index)
        context_input = self.dropout(base_embeddings, deterministic=not train, rngs=dropout_key)
        q_input = self.dropout(q_embedding, deterministic=not train, rngs=dropout_key)
        context_condition = self.context_to_hidden(maybe_cast(context_input, self.dtype)).astype(self.dtype)
        q_condition = self.q_to_hidden(maybe_cast(q_input, self.dtype)).astype(self.dtype)
        return context_condition, q_condition

    def _q_to_token_logits(self, q: Array, tokens: Array, step_index: Array) -> Array:
        del step_index
        return self._q_to_output_logits(q, tokens)

    def _q_update(
        self,
        tokens: Array,
        q_logits: Array,
        read_state: Array,
        step_index: Array,
    ) -> tuple[Array, dict[str, Array]]:
        del tokens
        current_logits = self._center_q_logits(q_logits)
        current_log_q = jax.nn.log_softmax(current_logits, axis=-1)
        current_q = jnp.exp(current_log_q)
        class_logits = self.q_proposal_head(maybe_cast(read_state, self.dtype)).astype(jnp.float32)
        proposal_log_q = jax.nn.log_softmax(class_logits, axis=-1)
        proposal_q = jnp.exp(proposal_log_q)
        flow_speed = jax.nn.sigmoid(
            self.q_update_head(maybe_cast(read_state, self.dtype)).astype(jnp.float32)
        )
        next_logits = (1.0 - flow_speed) * current_log_q + flow_speed * proposal_log_q
        next_logits = self._center_q_logits(next_logits)
        next_q = jax.nn.softmax(next_logits, axis=-1)

        proposal_tv_distance = 0.5 * jnp.sum(jnp.abs(proposal_q - current_q), axis=-1, keepdims=True)
        q_tv_delta = 0.5 * jnp.sum(jnp.abs(next_q - current_q), axis=-1, keepdims=True)
        kl_qp = jnp.sum(current_q * (current_log_q - proposal_log_q), axis=-1, keepdims=True)
        kl_pq = jnp.sum(proposal_q * (proposal_log_q - current_log_q), axis=-1, keepdims=True)
        symmetric_kl = 0.5 * (kl_qp + kl_pq)
        flow_kl_energy = jnp.square(flow_speed) * symmetric_kl
        diagnostics = {
            "flow_speed": flow_speed,
            "proposal_tv_distance": proposal_tv_distance,
            "q_tv_delta": q_tv_delta,
            "flow_kl_energy": flow_kl_energy,
        }
        return next_logits, diagnostics

    def _halt_logits(self, read_state: Array) -> Array:
        pooled = jnp.mean(read_state.astype(jnp.float32), axis=1)
        return self.halt_head(maybe_cast(pooled, self.dtype)).astype(jnp.float32)[..., 0]

    def _hidden_h_cycle(
        self,
        hidden_state: Array,
        context_condition: Array,
        q_condition: Array,
    ) -> Array:
        hidden = hidden_state.astype(self.dtype)
        # Within an h-cycle, cheap local propagation runs for every refine step.
        # The expensive all-to-all attention is reserved for the cycle boundary,
        # immediately before q commit / halt.
        for _ in range(self.refine_steps):
            for block in self.solver_blocks:
                hidden = block.local_think(hidden, context_condition, q_condition)
        for block in self.solver_blocks:
            hidden = block.global_communicate(
                hidden,
                context_condition,
                q_condition,
                self.rope_cos,
                self.rope_sin,
            )
        return hidden

    def _readout_fuse(
        self,
        hidden_state: Array,
        context_condition: Array,
        q_condition: Array,
    ) -> Array:
        hidden = self.readout_hidden_norm(hidden_state.astype(jnp.float32)).astype(jnp.float32)
        context_condition = self.readout_condition_norm(
            context_condition.astype(jnp.float32)
        ).astype(jnp.float32)
        base = self.readout_output_norm(hidden + context_condition).astype(self.dtype)
        return (base.astype(jnp.float32) + q_condition.astype(jnp.float32)).astype(self.dtype)

    def _q_step(
        self,
        tokens: Array,
        q: Array,
        q_history: Array,
        q_history_count: Array,
        hidden_state: Array,
        base_embeddings: Array,
        step_index: Array,
        *,
        train: bool,
        dropout_key: Array | None,
        stop_hidden_between_steps: bool = True,
    ) -> tuple[Array, Array, Array, Array]:
        halt_logits = jnp.zeros((tokens.shape[0],), dtype=jnp.float32)
        flow_diagnostics = {
            "flow_speed": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "proposal_tv_distance": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "q_tv_delta": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
            "flow_kl_energy": jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
        }
        context_condition, q_condition = self._typed_conditions(
            tokens,
            jax.lax.stop_gradient(q_history),
            jax.lax.stop_gradient(q_history_count),
            base_embeddings,
            step_index,
            train=train,
            dropout_key=dropout_key,
        )
        hidden_state = jax.lax.stop_gradient(hidden_state) if stop_hidden_between_steps else hidden_state
        hidden_state = self._hidden_h_cycle(
            hidden_state,
            context_condition,
            q_condition,
        )
        read_state = self._readout_fuse(
            hidden_state,
            context_condition,
            q_condition,
        )
        q, flow_diagnostics = self._q_update(
            tokens,
            q,
            read_state,
            step_index,
        )
        halt_logits = self._halt_logits(read_state)
        return (
            q,
            hidden_state,
            halt_logits,
            flow_diagnostics,
        )

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
            "q_history": self.initial_q_history(batch["inputs"]),
            "q_history_count": self.initial_q_history_count(batch["inputs"]),
            "hidden": jnp.zeros(
                (batch_size, self.config.seq_len, self.hidden_dim),
                dtype=self.dtype,
            ),
            "steps": jnp.zeros((batch_size,), dtype=jnp.int32),
            "halted": jnp.ones((batch_size,), dtype=bool),
            "current_inputs": jnp.zeros_like(batch["inputs"]),
            "current_labels": jnp.zeros_like(batch["labels"]),
            "current_example_mask": jnp.zeros((batch_size,), dtype=jnp.float32),
        }

    def forward_act_step(
        self,
        carry: dict[str, Array],
        batch: dict[str, Array],
        *,
        train: bool,
        dropout_key: Array | None = None,
        enable_halt: bool = True,
    ) -> tuple[dict[str, Array], Array, dict[str, Array]]:
        if dropout_key is None:
            dropout_key = jax.random.key(0)
        reset = carry["halted"]
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
        reset_q_history = self.initial_q_history(inputs, reset_q)
        reset_q_history_count = self.initial_q_history_count(inputs)
        q_history = jnp.where(reset[:, None, None, None], reset_q_history, carry["q_history"])
        q_history_count = jnp.where(reset, reset_q_history_count, carry["q_history_count"])
        reset_hidden = self.initial_hidden_state(
            inputs,
            reset_q,
            base_embeddings,
            train=train,
            dropout_key=dropout_key,
        )
        hidden = jnp.where(reset_state, reset_hidden, carry["hidden"])
        next_q, next_hidden, halt_logits, flow_diagnostics = self._q_step(
            inputs,
            q,
            q_history,
            q_history_count,
            hidden,
            base_embeddings,
            step_index,
            train=train,
            dropout_key=dropout_key,
            stop_hidden_between_steps=True,
        )
        logits = self._q_to_token_logits(next_q, inputs, step_index)
        new_steps = steps + 1
        next_q_history, next_q_history_count = self._append_q_history(
            q_history,
            next_q,
            q_history_count,
        )
        is_last_step = new_steps >= self.total_steps
        if train and enable_halt:
            halted = is_last_step | (halt_logits > 0.0)
            min_halt_steps = min(int(self.brc.halt_min_steps), self.total_steps)
            min_steps = jnp.full_like(new_steps, min_halt_steps)
            if self.total_steps > 1 and self.brc.halt_exploration_prob > 0.0:
                explore_key, min_step_key = jax.random.split(dropout_key)
                explore = jax.random.uniform(explore_key, halt_logits.shape) < self.brc.halt_exploration_prob
                random_step = jax.random.randint(
                    min_step_key,
                    halt_logits.shape,
                    min_halt_steps,
                    self.total_steps + 1,
                )
                min_steps = jnp.where(explore, random_step, min_steps)
            halted = halted & (new_steps >= min_steps)
        else:
            halted = is_last_step
        new_carry = {
            "q": jax.lax.stop_gradient(next_q),
            "q_history": jax.lax.stop_gradient(next_q_history),
            "q_history_count": jax.lax.stop_gradient(next_q_history_count),
            "hidden": jax.lax.stop_gradient(next_hidden),
            "steps": jax.lax.stop_gradient(new_steps),
            "halted": jax.lax.stop_gradient(halted),
            "current_inputs": inputs,
            "current_labels": labels,
            "current_example_mask": example_mask,
        }
        diagnostics = {
            "halt_logits": halt_logits,
            "act_step": jnp.mean(new_steps.astype(jnp.float32)),
            "halted_rate": jnp.mean(halted.astype(jnp.float32)),
            "reset_rate": jnp.mean(reset.astype(jnp.float32)),
            "flow_speed": jnp.mean(flow_diagnostics["flow_speed"].astype(jnp.float32)),
            "proposal_tv_distance": jnp.mean(flow_diagnostics["proposal_tv_distance"].astype(jnp.float32)),
            "q_tv_delta": jnp.mean(flow_diagnostics["q_tv_delta"].astype(jnp.float32)),
            "flow_kl_energy": jnp.mean(flow_diagnostics["flow_kl_energy"].astype(jnp.float32)),
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
        initial_q_history = self.initial_q_history(tokens, initial_q)
        initial_q_history_count = self.initial_q_history_count(tokens)
        base_embeddings, context = self.context_memory(tokens)
        query_mask = (~context).astype(jnp.float32)
        query_normalizer = jnp.maximum(jnp.sum(query_mask), 1.0)

        def scan_step(carry, scan_inputs):
            step_index, step_dropout_key = scan_inputs
            q, q_history, q_history_count, hidden_state = carry
            next_q, next_hidden, halt_logits, flow_diagnostics = self._q_step(
                tokens,
                q,
                q_history,
                q_history_count,
                hidden_state,
                base_embeddings,
                step_index,
                train=train,
                dropout_key=step_dropout_key,
                stop_hidden_between_steps=True,
            )
            next_q_history, next_q_history_count = self._append_q_history(
                q_history,
                next_q,
                q_history_count,
            )
            next_carry = (next_q, next_q_history, next_q_history_count, next_hidden)
            if return_final_only:
                return next_carry, None
            logits = self._q_to_token_logits(next_q, tokens, step_index)
            confidence = jnp.max(self._normalize_q(next_q), axis=-1)
            q_top1_probability = jnp.sum(confidence * query_mask) / query_normalizer
            return next_carry, (
                logits,
                q_top1_probability,
                halt_logits,
                jnp.mean(flow_diagnostics["flow_speed"].astype(jnp.float32)),
                jnp.mean(flow_diagnostics["proposal_tv_distance"].astype(jnp.float32)),
                jnp.mean(flow_diagnostics["q_tv_delta"].astype(jnp.float32)),
                jnp.mean(flow_diagnostics["flow_kl_energy"].astype(jnp.float32)),
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
            initial_q_history.astype(jnp.float32),
            initial_q_history_count,
            initial_hidden,
        )
        final_carry, scan_outputs = jax.lax.scan(
            scan_step,
            initial_carry,
            (step_indices, step_dropout_keys),
        )
        q_final, _q_history_final, _q_history_count_final, _hidden_final = final_carry
        if return_final_only:
            final_step = jnp.asarray(self.total_steps - 1, dtype=jnp.int32)
            logits = self._q_to_token_logits(q_final, tokens, final_step)
            diagnostics = {
                "q_top1_probability": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "unroll_steps": jnp.asarray(self.total_steps, dtype=jnp.float32),
                "halt_logits": jnp.zeros((self.total_steps, tokens.shape[0]), dtype=jnp.float32),
                "flow_speed": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "proposal_tv_distance": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "q_tv_delta": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "flow_kl_energy": jnp.zeros((self.total_steps,), dtype=jnp.float32),
            }
            return logits, diagnostics
        (
            step_logits,
            q_top1_probability,
            halt_logits,
            flow_speed,
            proposal_tv_distance,
            q_tv_delta,
            flow_kl_energy,
        ) = scan_outputs
        diagnostics = {
            "q_top1_probability": q_top1_probability,
            "halt_logits": halt_logits,
            "unroll_steps": jnp.asarray(self.total_steps, dtype=jnp.float32),
            "flow_speed": flow_speed,
            "proposal_tv_distance": proposal_tv_distance,
            "q_tv_delta": q_tv_delta,
            "flow_kl_energy": flow_kl_energy,
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
