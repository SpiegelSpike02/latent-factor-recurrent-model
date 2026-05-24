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
        self.local_norm = unscaled_rms_norm(hidden_dim, brc.rms_norm_eps, dtype, rngs)
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
        input_scale = float(self.config.brc_config.input_scale)
        route_condition = input_scale * context_condition.astype(jnp.float32)
        q_hint = q_condition.astype(jnp.float32)
        route = self.attn_norm(h.astype(jnp.float32) + route_condition).astype(self.dtype)
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
        h = (h.astype(jnp.float32) + self.attn_scale * attn.astype(jnp.float32)).astype(self.dtype)
        local_base = self.local_norm(h.astype(jnp.float32) + route_condition).astype(self.dtype)
        local_input = (local_base.astype(jnp.float32) + q_hint).astype(self.dtype)
        local = self.local_mlp(local_input).astype(jnp.float32)
        return (h.astype(jnp.float32) + self.local_scale * local).astype(self.dtype)


class BRCModel(nnx.Module):
    """Q-state recurrent solver for fixed-size grid reasoning tasks.

    The recurrent state is the per-cell answer q on the class simplex.
    Scheduled gamma controls how strongly q is read by the hidden workspace;
    BRC does not learn a separate per-cell confidence state.
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
        if config.task_type == "sudoku" and config.vocab_size < 11:
            raise ValueError("BRC Sudoku expects vocab_size >= 11")
        brc = config.brc_config
        if brc.q_steps < 1:
            raise ValueError("BRC q_steps must be at least 1")
        if min(brc.h_steps, brc.block_depth) < 1:
            raise ValueError("BRC h_steps and block_depth must be at least 1")
        if not 0.0 <= brc.gamma < 1.0:
            raise ValueError("BRC gamma must be in [0, 1)")
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
        self.q_steps = int(brc.q_steps)
        self.h_steps = int(brc.h_steps)
        self.hidden_dim = hidden_dim
        self.dtype = compute_dtype(runtime.compute_dtype)
        self.embed_scale = math.sqrt(config.d_model)
        self.q_vocab_size = config.vocab_size
        self.q_eps = 1e-6
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
            config.vocab_size,
            config.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            embedding_init=embed_init,
            rngs=rngs,
        )
        self.context_embed = nnx.Embed(2, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
        self.q_embed = nnx.Embed(
            max(10, config.vocab_size),
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
            config.vocab_size,
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
            return tokens > 1
        return tokens != 0

    def _normalize_q(self, q: Array) -> Array:
        q = jnp.maximum(q.astype(jnp.float32), self.q_eps)
        return q / jnp.sum(q, axis=-1, keepdims=True)

    def _uniform_q(self, tokens: Array) -> Array:
        return jnp.full(
            (*tokens.shape, self.q_vocab_size),
            1.0 / float(self.q_vocab_size),
            dtype=jnp.float32,
        )

    def _q_to_output_logits(self, q: Array, tokens: Array) -> Array:
        del tokens
        q = self._normalize_q(q)
        return jnp.log(jnp.maximum(q, self.output_logit_eps))

    def initial_q(self, tokens: Array) -> Array:
        return self._uniform_q(tokens)

    def _position_embeddings(self) -> Array:
        if self.brc.position_encoding != "learned":
            return jnp.zeros((self.config.seq_len, self.config.d_model), dtype=self.dtype)
        return (
            self.row_embed(self.row_ids)
            + self.col_embed(self.col_ids)
            + self.box_embed(self.box_ids)
        )

    def _q_embedding(self, tokens: Array, q: Array, step_index: Array) -> Array:
        del tokens
        q = self._normalize_q(q)
        uniform = jnp.full_like(q, 1.0 / float(self.q_vocab_size))
        gamma = self._trust_gamma(step_index)
        if gamma.ndim == 0:
            gamma = gamma[None, None, None]
        elif gamma.ndim == 1:
            gamma = gamma[:, None, None]
        trusted_q = gamma * (q - uniform)
        embedding_table = maybe_cast(self.q_embed.embedding[: self.config.vocab_size], self.dtype)
        q_embedding = jnp.einsum(
            "bnd,dk->bnk",
            maybe_cast(trusted_q, self.dtype),
            embedding_table,
            preferred_element_type=jnp.float32,
        )
        return q_embedding

    def _typed_conditions(
        self,
        tokens: Array,
        q: Array,
        base_embeddings: Array,
        step_index: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> tuple[Array, Array]:
        q_embedding = self._q_embedding(tokens, q, step_index)
        context_input = self.dropout(base_embeddings, deterministic=not train, rngs=dropout_key)
        q_input = self.dropout(q_embedding, deterministic=not train, rngs=dropout_key)
        context_condition = self.context_to_hidden(maybe_cast(context_input, self.dtype)).astype(self.dtype)
        q_condition = self.q_to_hidden(maybe_cast(q_input, self.dtype)).astype(self.dtype)
        return context_condition, q_condition

    def _q_to_token_logits(self, q: Array, tokens: Array, step_index: Array) -> Array:
        del step_index
        return self._q_to_output_logits(q, tokens)

    def _trust_gamma(self, step_index: Array) -> Array:
        progress = (step_index.astype(jnp.float32) + 1.0) / float(self.q_steps)
        return float(self.brc.gamma) * jnp.square(progress)

    def _q_update(self, tokens: Array, q: Array, read_state: Array, step_index: Array) -> tuple[Array, Array]:
        del tokens
        current_q = self._normalize_q(q)
        class_logits = self.q_proposal_head(maybe_cast(read_state, self.dtype)).astype(jnp.float32)
        proposal_q = jax.nn.softmax(class_logits, axis=-1)
        update_alpha = jax.nn.sigmoid(
            self.q_update_head(maybe_cast(read_state, self.dtype)).astype(jnp.float32)
        )
        next_q = (1.0 - update_alpha) * current_q + update_alpha * proposal_q
        next_q = self._normalize_q(next_q)
        return next_q, update_alpha

    def _halt_logits(self, read_state: Array) -> Array:
        pooled = jnp.mean(read_state.astype(jnp.float32), axis=1)
        return self.halt_head(maybe_cast(pooled, self.dtype)).astype(jnp.float32)[..., 0]

    def _hidden_l_cycle(
        self,
        hidden_state: Array,
        context_condition: Array,
        q_condition: Array,
    ) -> Array:
        hidden = hidden_state.astype(self.dtype)
        for _ in range(self.h_steps):
            for block in self.solver_blocks:
                hidden = block(
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
        input_scale = float(self.brc.input_scale)
        hidden = self.readout_hidden_norm(hidden_state.astype(jnp.float32)).astype(self.dtype)
        context_condition = self.readout_condition_norm(
            input_scale * context_condition.astype(jnp.float32)
        ).astype(self.dtype)
        base = self.readout_output_norm(
            hidden.astype(jnp.float32) + context_condition.astype(jnp.float32)
        ).astype(self.dtype)
        return (base.astype(jnp.float32) + q_condition.astype(jnp.float32)).astype(self.dtype)

    def _q_step(
        self,
        tokens: Array,
        q: Array,
        hidden_state: Array,
        base_embeddings: Array,
        step_index: Array,
        *,
        train: bool,
        dropout_key: Array | None,
        stop_hidden_between_steps: bool = True,
    ) -> tuple[Array, Array, Array, Array]:
        halt_logits = jnp.zeros((tokens.shape[0],), dtype=jnp.float32)
        q_update_alpha = jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32)
        context_condition, q_condition = self._typed_conditions(
            tokens,
            jax.lax.stop_gradient(q),
            base_embeddings,
            step_index,
            train=train,
            dropout_key=dropout_key,
        )
        hidden_state = jax.lax.stop_gradient(hidden_state) if stop_hidden_between_steps else hidden_state
        hidden_state = self._hidden_l_cycle(
            hidden_state,
            context_condition,
            q_condition,
        )
        read_state = self._readout_fuse(
            hidden_state,
            context_condition,
            q_condition,
        )
        q, q_update_alpha = self._q_update(
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
            q_update_alpha,
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
        base_embeddings = (
            self.puzzle_embed(tokens.astype(jnp.int32))
            + self.context_embed(context.astype(jnp.int32))
        )
        if self.brc.position_encoding == "learned":
            base_embeddings = base_embeddings + self._position_embeddings()[None, :, :]
        return base_embeddings, context

    def initial_carry(self, batch: dict[str, Array]) -> dict[str, Array]:
        batch_size = batch["inputs"].shape[0]
        return {
            "q": self._uniform_q(batch["inputs"]),
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
        step_index = jnp.minimum(steps, self.q_steps - 1)
        reset_q = self.initial_q(inputs)
        q = jnp.where(reset_state, reset_q, carry["q"])
        reset_hidden = self.initial_hidden_state(
            inputs,
            reset_q,
            base_embeddings,
            train=train,
            dropout_key=dropout_key,
        )
        hidden = jnp.where(reset_state, reset_hidden, carry["hidden"])
        next_q, next_hidden, halt_logits, q_update_alpha = self._q_step(
            inputs,
            q,
            hidden,
            base_embeddings,
            step_index,
            train=train,
            dropout_key=dropout_key,
            stop_hidden_between_steps=True,
        )
        logits = self._q_to_token_logits(next_q, inputs, step_index)
        new_steps = steps + 1
        is_last_step = new_steps >= self.q_steps
        if train:
            halted = is_last_step | (halt_logits > 0.0)
            min_halt_steps = min(int(self.brc.halt_min_steps), self.q_steps)
            min_steps = jnp.full_like(new_steps, min_halt_steps)
            if self.q_steps > 1 and self.brc.halt_exploration_prob > 0.0:
                explore_key, min_step_key = jax.random.split(dropout_key)
                explore = jax.random.uniform(explore_key, halt_logits.shape) < self.brc.halt_exploration_prob
                random_step = jax.random.randint(
                    min_step_key,
                    halt_logits.shape,
                    min_halt_steps,
                    self.q_steps + 1,
                )
                min_steps = jnp.where(explore, random_step, min_steps)
            halted = halted & (new_steps >= min_steps)
        else:
            halted = is_last_step
        new_carry = {
            "q": jax.lax.stop_gradient(next_q),
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
            "q_update_alpha": jnp.mean(q_update_alpha.astype(jnp.float32)),
        }
        return new_carry, logits, diagnostics

    def run_q_steps(
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
        base_embeddings, context = self.context_memory(tokens)
        query_mask = (~context).astype(jnp.float32)
        query_normalizer = jnp.maximum(jnp.sum(query_mask), 1.0)

        def scan_step(carry, scan_inputs):
            step_index, step_dropout_key = scan_inputs
            q, hidden_state = carry
            next_q, next_hidden, halt_logits, q_update_alpha = self._q_step(
                tokens,
                q,
                hidden_state,
                base_embeddings,
                step_index,
                train=train,
                dropout_key=step_dropout_key,
                stop_hidden_between_steps=True,
            )
            next_carry = (next_q, next_hidden)
            if return_final_only:
                return next_carry, None
            logits = self._q_to_token_logits(next_q, tokens, step_index)
            confidence = jnp.max(self._normalize_q(next_q), axis=-1)
            q_confidence = jnp.sum(confidence * query_mask) / query_normalizer
            return next_carry, (
                logits,
                q_confidence,
                halt_logits,
                jnp.mean(q_update_alpha.astype(jnp.float32)),
            )

        step_indices = jnp.arange(self.q_steps, dtype=jnp.int32)
        if dropout_key is None:
            step_dropout_keys = jax.random.split(jax.random.key(0), self.q_steps)
        else:
            step_dropout_keys = jax.random.split(dropout_key, self.q_steps)
        initial_hidden = self.initial_hidden_state(
            tokens,
            initial_q,
            base_embeddings,
            train=train,
            dropout_key=step_dropout_keys[0],
        )
        initial_carry = (initial_q.astype(jnp.float32), initial_hidden)
        final_carry, scan_outputs = jax.lax.scan(
            scan_step,
            initial_carry,
            (step_indices, step_dropout_keys),
        )
        q_final, _hidden_final = final_carry
        if return_final_only:
            final_step = jnp.asarray(self.q_steps - 1, dtype=jnp.int32)
            logits = self._q_to_token_logits(q_final, tokens, final_step)
            diagnostics = {
                "q_confidence": jnp.zeros((self.q_steps,), dtype=jnp.float32),
                "unroll_steps": jnp.asarray(self.q_steps, dtype=jnp.float32),
                "halt_logits": jnp.zeros((self.q_steps, tokens.shape[0]), dtype=jnp.float32),
                "q_update_alpha": jnp.zeros((self.q_steps,), dtype=jnp.float32),
            }
            return logits, diagnostics
        (
            step_logits,
            q_confidence,
            halt_logits,
            q_update_alpha,
        ) = scan_outputs
        diagnostics = {
            "q_confidence": q_confidence,
            "halt_logits": halt_logits,
            "unroll_steps": jnp.asarray(self.q_steps, dtype=jnp.float32),
            "q_update_alpha": q_update_alpha,
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
        return self.run_q_steps(
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
