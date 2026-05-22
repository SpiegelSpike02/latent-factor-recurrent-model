from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from lfrm.config import ModelConfig, RuntimeConfig
from .common import Array, casted_linear_init, compute_dtype, maybe_cast, trunc_normal_init
from .recurrent.layers import SwiGLU, dot_product_attention, multi_head_attention_with_rope, unscaled_rms_norm


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
        self.self_norm = unscaled_rms_norm(hidden_dim, config.brc_config.rms_norm_eps, dtype, rngs)
        self.message_norm = unscaled_rms_norm(hidden_dim, config.brc_config.rms_norm_eps, dtype, rngs)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.channel_mlp = SwiGLU(
            hidden_dim,
            mlp_ratio,
            dtype=dtype,
            rngs=rngs,
        )
    def __call__(
        self,
        h: Array,
        rope_cos: Array | None,
        rope_sin: Array | None,
    ) -> Array:
        attn = multi_head_attention_with_rope(
            self.self_attention,
            h,
            h,
            h,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            name="BRC self attention",
        )
        h = self.self_norm(h.astype(jnp.float32) + attn.astype(jnp.float32)).astype(self.dtype)
        msg = self.channel_mlp(maybe_cast(h, self.dtype)).astype(jnp.float32)
        return self.message_norm(h.astype(jnp.float32) + msg).astype(self.dtype)


class BRCModel(nnx.Module):
    """Belief recurrent controller for fixed-size grid reasoning tasks."""

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
        if brc.evidence_steps < 1:
            raise ValueError("BRC evidence_steps must be at least 1")
        if min(brc.h_cycles, brc.l_cycles, brc.l_layers) < 1:
            raise ValueError("BRC h_cycles, l_cycles, and l_layers must be at least 1")
        hidden_dim = int(brc.hidden_state_dim) if brc.hidden_state_dim > 0 else config.d_model
        if hidden_dim < 1:
            raise ValueError("BRC hidden_state_dim must be positive or 0 for d_model")
        if brc.num_heads < 1:
            raise ValueError("BRC num_heads must be at least 1")
        if hidden_dim % brc.num_heads != 0:
            raise ValueError("BRC hidden state dimension must be divisible by num_heads")
        if brc.mlp_ratio < 1:
            raise ValueError("BRC mlp_ratio must be at least 1")
        if brc.position_encoding not in ("rope", "learned", "none"):
            raise ValueError("BRC position_encoding must be 'rope', 'learned', or 'none'")
        if brc.position_encoding == "rope" and (hidden_dim // brc.num_heads) % 2 != 0:
            raise ValueError("BRC RoPE head dimension must be even")
        if brc.step_loss_schedule not in ("uniform", "linear"):
            raise ValueError("BRC step_loss_schedule must be 'uniform' or 'linear'")

        self.config = config
        self.runtime = runtime
        self.brc = brc
        self.evidence_steps = int(brc.evidence_steps)
        self.h_cycles = int(brc.h_cycles)
        self.l_cycles = int(brc.l_cycles)
        self.hidden_dim = hidden_dim
        self.dtype = compute_dtype(runtime.compute_dtype)
        self.embed_scale = math.sqrt(config.d_model)
        self.belief_vocab_size = config.vocab_size
        self.evidence_eps = 1e-4
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
        self.belief_embed = nnx.Embed(
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
        self.time_embed = nnx.Embed(self.evidence_steps, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        if brc.position_encoding == "rope":
            head_dim = self.hidden_dim // brc.num_heads
            inv_freq = 1.0 / (brc.rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
            positions = jnp.arange(config.seq_len, dtype=jnp.float32)
            freqs = positions[:, None] * inv_freq[None, :]
            rope = jnp.concatenate((freqs, freqs), axis=-1)
            self.rope_cos = nnx.data(jnp.cos(rope))
            self.rope_sin = nnx.data(jnp.sin(rope))
        else:
            self.rope_cos = None
            self.rope_sin = None

        self.input_to_hidden = nnx.Linear(
            config.d_model,
            self.hidden_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.hidden_anchor_norm = unscaled_rms_norm(self.hidden_dim, brc.rms_norm_eps, self.dtype, rngs)
        self.readout_hidden_norm = unscaled_rms_norm(self.hidden_dim, brc.rms_norm_eps, self.dtype, rngs)
        self.readout_condition_norm = unscaled_rms_norm(self.hidden_dim, brc.rms_norm_eps, self.dtype, rngs)
        self.readout_output_norm = unscaled_rms_norm(self.hidden_dim, brc.rms_norm_eps, self.dtype, rngs)
        self.readout_condition = nnx.Linear(
            config.d_model,
            self.hidden_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
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
                for _ in range(brc.l_layers)
            ]
        )
        self.lm_head = nnx.Linear(
            self.hidden_dim,
            config.vocab_size,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
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

    def _evidence_alpha(self, evidence: Array) -> Array:
        return jnp.maximum(evidence.astype(jnp.float32), self.evidence_eps)

    def _evidence_probabilities(self, evidence: Array) -> tuple[Array, Array, Array]:
        alpha = self._evidence_alpha(evidence)
        strength = jnp.sum(alpha, axis=-1, keepdims=True)
        probabilities = alpha / jnp.maximum(strength, self.evidence_eps)
        uncertainty = jnp.minimum(self.belief_vocab_size / jnp.maximum(strength, self.evidence_eps), 1.0)
        return probabilities, strength, uncertainty

    def _evidence_to_output_logits(self, evidence: Array, tokens: Array) -> Array:
        del tokens
        return jnp.log(self._evidence_alpha(evidence))

    def initial_evidence(self, tokens: Array) -> Array:
        return jnp.ones((*tokens.shape, self.belief_vocab_size), dtype=jnp.float32)

    def _position_embeddings(self) -> Array:
        if self.brc.position_encoding != "learned":
            return jnp.zeros((self.config.seq_len, self.config.d_model), dtype=self.dtype)
        return (
            self.row_embed(self.row_ids)
            + self.col_embed(self.col_ids)
            + self.box_embed(self.box_ids)
        )

    def _evidence_embedding(self, tokens: Array, evidence: Array) -> Array:
        del tokens
        evidence_probs, _strength, uncertainty = self._evidence_probabilities(evidence)
        embedding_table = maybe_cast(self.belief_embed.embedding[: self.config.vocab_size], self.dtype)
        evidence_embedding = jnp.einsum(
            "bnd,dk->bnk",
            maybe_cast(evidence_probs, self.dtype),
            embedding_table,
            preferred_element_type=jnp.float32,
        )
        reliability = (1.0 - uncertainty).astype(evidence_embedding.dtype)
        return evidence_embedding * reliability

    def _cell_embeddings(
        self,
        tokens: Array,
        evidence: Array,
        base_embeddings: Array,
        time_embedding: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> Array:
        evidence_embedding = self._evidence_embedding(tokens, evidence)
        time = time_embedding[None, None, :] if time_embedding.ndim == 1 else time_embedding[:, None, :]
        x = (
            base_embeddings
            + evidence_embedding
            + time
        ) * math.sqrt(1.0 / 3.0)
        return self.dropout(x, deterministic=not train, rngs=dropout_key).astype(self.dtype)

    def _evidence_to_token_logits(self, evidence: Array, tokens: Array, step_index: Array) -> Array:
        del step_index
        return self._evidence_to_output_logits(evidence, tokens)

    def _evidence_update(self, tokens: Array, evidence: Array, proposal_logits: Array, step_index: Array) -> Array:
        del tokens, step_index
        evidence_delta = jax.nn.softplus(proposal_logits.astype(jnp.float32)) + self.evidence_eps
        return self._evidence_alpha(evidence) + evidence_delta

    def _halt_logits(self, read_state: Array) -> Array:
        pooled = jnp.mean(read_state.astype(jnp.float32), axis=1)
        return self.halt_head(maybe_cast(pooled, self.dtype)).astype(jnp.float32)[..., 0]

    def _hidden_l_cycle(self, hidden_state: Array, hidden_input: Array) -> Array:
        hidden = hidden_state.astype(self.dtype)
        for _ in range(self.l_cycles):
            hidden = self.hidden_anchor_norm(hidden.astype(jnp.float32) + hidden_input.astype(jnp.float32)).astype(self.dtype)
            for block in self.solver_blocks:
                hidden = block(hidden, self.rope_cos, self.rope_sin)
        return hidden

    def _readout_fuse(self, hidden_state: Array, cell_input: Array) -> Array:
        hidden = self.readout_hidden_norm(hidden_state.astype(jnp.float32)).astype(self.dtype)
        condition = self.readout_condition(maybe_cast(cell_input, self.dtype)).astype(self.dtype)
        condition = self.readout_condition_norm(condition.astype(jnp.float32)).astype(self.dtype)
        fused = hidden.astype(jnp.float32) + condition.astype(jnp.float32)
        return self.readout_output_norm(fused).astype(self.dtype)

    def _evidence_step(
        self,
        tokens: Array,
        evidence: Array,
        hidden_state: Array,
        base_embeddings: Array,
        time_embedding: Array,
        step_index: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> tuple[Array, Array, Array]:
        halt_logits = jnp.zeros((tokens.shape[0],), dtype=jnp.float32)
        for h_index in range(self.h_cycles):
            cell_input = self._cell_embeddings(
                tokens,
                evidence,
                base_embeddings,
                time_embedding,
                train=train,
                dropout_key=dropout_key,
            )
            hidden_input = self.input_to_hidden(maybe_cast(cell_input, self.dtype)).astype(self.dtype)
            hidden_state = self._hidden_l_cycle(hidden_state, hidden_input)
            read_state = self._readout_fuse(hidden_state, cell_input)
            proposal_logits = self.lm_head(maybe_cast(read_state, self.dtype))
            evidence = self._evidence_update(tokens, evidence, proposal_logits, step_index)
            halt_logits = self._halt_logits(read_state)
            if h_index < self.h_cycles - 1:
                hidden_state = jax.lax.stop_gradient(hidden_state)
        return evidence, jax.lax.stop_gradient(hidden_state), halt_logits

    def initial_hidden_state(
        self,
        tokens: Array,
        evidence: Array,
        base_embeddings: Array,
        time_embedding: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> Array:
        del evidence, base_embeddings, time_embedding, train, dropout_key
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
            "evidence": jnp.ones(
                (batch_size, self.config.seq_len, self.belief_vocab_size),
                dtype=jnp.float32,
            ),
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
        step_index = jnp.minimum(steps, self.evidence_steps - 1)
        time_embedding = self.time_embed(step_index)
        reset_evidence = self.initial_evidence(inputs)
        evidence = jnp.where(reset_state, reset_evidence, carry["evidence"])
        reset_hidden = self.initial_hidden_state(
            inputs,
            reset_evidence,
            base_embeddings,
            time_embedding,
            train=train,
            dropout_key=dropout_key,
        )
        hidden = jnp.where(reset_state, reset_hidden, carry["hidden"])
        next_evidence, next_hidden, halt_logits = self._evidence_step(
            inputs,
            evidence,
            hidden,
            base_embeddings,
            time_embedding,
            step_index,
            train=train,
            dropout_key=dropout_key,
        )
        logits = self._evidence_to_token_logits(next_evidence, inputs, step_index)
        new_steps = steps + 1
        is_last_step = new_steps >= self.evidence_steps
        if train:
            halted = is_last_step | (halt_logits > 0.0)
            if self.evidence_steps > 1 and self.brc.halt_exploration_prob > 0.0:
                explore_key, min_step_key = jax.random.split(dropout_key)
                explore = jax.random.uniform(explore_key, halt_logits.shape) < self.brc.halt_exploration_prob
                random_step = jax.random.randint(
                    min_step_key,
                    halt_logits.shape,
                    2,
                    self.evidence_steps + 1,
                )
                min_steps = jnp.where(explore, random_step, 1)
                halted = halted & (new_steps >= min_steps)
        else:
            halted = is_last_step
        new_carry = {
            "evidence": jax.lax.stop_gradient(next_evidence),
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
        }
        return new_carry, logits, diagnostics

    def run_diffusion(
        self,
        tokens: Array,
        *,
        initial_evidence: Array | None = None,
        train: bool,
        dropout_key: Array | None = None,
        return_final_only: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        if initial_evidence is None:
            initial_evidence = self.initial_evidence(tokens)
        base_embeddings, context = self.context_memory(tokens)
        query_mask = (~context).astype(jnp.float32)
        query_normalizer = jnp.maximum(jnp.sum(query_mask), 1.0)

        def scan_step(carry, scan_inputs):
            step_index, step_dropout_key, time_embedding = scan_inputs
            evidence, hidden_state = carry
            next_evidence, next_hidden, halt_logits = self._evidence_step(
                tokens,
                evidence,
                hidden_state,
                base_embeddings,
                time_embedding,
                step_index,
                train=train,
                dropout_key=step_dropout_key,
            )
            next_carry = (next_evidence, next_hidden)
            if return_final_only:
                return next_carry, None
            logits = self._evidence_to_token_logits(next_evidence, tokens, step_index)
            evidence_probs, _strength, _uncertainty = self._evidence_probabilities(next_evidence)
            confidence = jnp.max(evidence_probs, axis=-1)
            filled_ratio = jnp.sum(confidence * query_mask) / query_normalizer
            return next_carry, (
                logits,
                filled_ratio,
                halt_logits,
            )

        step_indices = jnp.arange(self.evidence_steps, dtype=jnp.int32)
        if dropout_key is None:
            step_dropout_keys = jax.random.split(jax.random.key(0), self.evidence_steps)
        else:
            step_dropout_keys = jax.random.split(dropout_key, self.evidence_steps)
        time_embeddings = self.time_embed(step_indices)
        initial_hidden = self.initial_hidden_state(
            tokens,
            initial_evidence,
            base_embeddings,
            time_embeddings[0],
            train=train,
            dropout_key=step_dropout_keys[0],
        )
        initial_carry = (initial_evidence.astype(jnp.float32), initial_hidden)
        final_carry, scan_outputs = jax.lax.scan(
            scan_step,
            initial_carry,
            (step_indices, step_dropout_keys, time_embeddings),
        )
        evidence_final, _hidden_final = final_carry
        if return_final_only:
            final_step = jnp.asarray(self.evidence_steps - 1, dtype=jnp.int32)
            logits = self._evidence_to_token_logits(evidence_final, tokens, final_step)
            diagnostics = {
                "diffusion_filled_ratio": jnp.zeros((self.evidence_steps,), dtype=jnp.float32),
                "unroll_steps": jnp.asarray(self.evidence_steps, dtype=jnp.float32),
                "halt_logits": jnp.zeros((self.evidence_steps, tokens.shape[0]), dtype=jnp.float32),
            }
            return logits, diagnostics
        step_logits, filled_ratio, halt_logits = scan_outputs
        diagnostics = {
            "diffusion_filled_ratio": filled_ratio,
            "halt_logits": halt_logits,
            "unroll_steps": jnp.asarray(self.evidence_steps, dtype=jnp.float32),
        }
        return step_logits, diagnostics

    def forward_all_steps_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        initial_evidence: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        return self.run_diffusion(
            tokens,
            initial_evidence=initial_evidence,
            train=train,
            dropout_key=dropout_key,
        )

    def forward_final_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        initial_evidence: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        return self.forward_all_steps_with_diagnostics(
            tokens,
            train=train,
            dropout_key=dropout_key,
            initial_evidence=initial_evidence,
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
