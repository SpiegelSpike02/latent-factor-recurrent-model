from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from lfrm.config import ModelConfig, RuntimeConfig
from lfrm.models.common import (
    Array,
    casted_linear_init,
    compute_dtype,
    gather_embedding_rows,
    maybe_cast,
    trunc_normal_init,
)
from lfrm.models.grid_layers import (
    CastedEmbedding,
    ConvSwiGLU2D,
    FullAttention,
    build_2d_axial_rope,
    run_truncated_h_cycles,
    unscaled_rms_norm,
)


class BeliefDynamicsBlock(nnx.Module):
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
        self.self_attention = FullAttention(
            hidden_dim,
            num_heads,
            dtype,
            name="BDR",
            rngs=rngs,
        )
        bdr = config.bdr_config
        self.attn_norm = unscaled_rms_norm(hidden_dim, bdr.rms_norm_eps, dtype, rngs)
        self.local_norm = unscaled_rms_norm(hidden_dim, bdr.rms_norm_eps, dtype, rngs)
        self.attn_scale = float(bdr.attn_scale)
        self.local_scale = float(bdr.local_scale)
        self.local_mlp = ConvSwiGLU2D(
            hidden_dim,
            mlp_ratio,
            bdr.local_kernel,
            config.grid_height,
            config.grid_width,
            dtype,
            min_intermediate_size=hidden_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        h: Array,
        context_condition: Array,
        draft_condition: Array,
        rope_cos: Array | None,
        rope_sin: Array | None,
    ) -> Array:
        h = self.global_communicate(
            h,
            context_condition,
            draft_condition,
            rope_cos,
            rope_sin,
        )
        return self.local_think(h, context_condition, draft_condition)

    def _self_attention(
        self,
        hidden_state: Array,
        rope_cos: Array | None,
        rope_sin: Array | None,
    ) -> Array:
        return self.self_attention(
            hidden_state,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

    def global_communicate(
        self,
        h: Array,
        context_condition: Array,
        draft_condition: Array,
        rope_cos: Array | None,
        rope_sin: Array | None,
    ) -> Array:
        injected = (
            h.astype(jnp.float32)
            + context_condition.astype(jnp.float32)
            + draft_condition.astype(jnp.float32)
        ).astype(self.dtype)
        attended_state = self.attn_norm(injected).astype(self.dtype)
        attn = self._self_attention(attended_state, rope_cos, rope_sin)
        return (injected.astype(jnp.float32) + self.attn_scale * attn.astype(jnp.float32)).astype(self.dtype)

    def local_think(
        self,
        h: Array,
        context_condition: Array,
        draft_condition: Array,
    ) -> Array:
        del context_condition, draft_condition
        local_input = self.local_norm(h.astype(jnp.float32)).astype(self.dtype)
        local = self.local_mlp(local_input).astype(jnp.float32)
        return (h.astype(jnp.float32) + self.local_scale * local).astype(self.dtype)


class BeliefDynamicsReasoner(nnx.Module):
    """Z-state recurrent solver for fixed-size grid reasoning tasks.

    The recurrent state stores centered answer logits ``z``. Its softmax is the
    per-cell explicit answer distribution view, which is read as a typed
    hypothesis hint; BDR does not learn a separate per-cell confidence state.
    """

    def __init__(
        self,
        config: ModelConfig,
        runtime: RuntimeConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if config.model_type != "bdr":
            raise ValueError("BeliefDynamicsReasoner requires model_type='bdr'")
        if config.grid_height * config.grid_width != config.seq_len:
            raise ValueError("grid_height * grid_width must equal seq_len")
        if config.task_type == "sudoku" and config.vocab_size != 9:
            raise ValueError("BDR Sudoku expects output vocab_size=9")
        bdr = config.bdr_config
        if bdr.commit_steps < 1:
            raise ValueError("BDR commit_steps must be at least 1")
        if min(bdr.h_cycles, bdr.l_cycles, bdr.l_layers) < 1:
            raise ValueError("BDR h_cycles, l_cycles, and l_layers must be at least 1")
        hidden_dim = int(bdr.hidden_state_dim) if bdr.hidden_state_dim > 0 else config.d_model
        if hidden_dim < 1:
            raise ValueError("BDR hidden_state_dim must be positive or 0 for d_model")
        if bdr.num_heads < 1:
            raise ValueError("BDR num_heads must be at least 1")
        if hidden_dim % bdr.num_heads != 0:
            raise ValueError("BDR hidden state dimension must be divisible by num_heads")
        if bdr.mlp_ratio < 1:
            raise ValueError("BDR mlp_ratio must be at least 1")
        if bdr.local_kernel < 1 or bdr.local_kernel % 2 == 0:
            raise ValueError("BDR local_kernel must be a positive odd integer")
        if bdr.position_encoding not in ("rope", "learned", "none"):
            raise ValueError("BDR position_encoding must be 'rope', 'learned', or 'none'")
        if bdr.position_encoding == "rope" and (hidden_dim // bdr.num_heads) % 4 != 0:
            raise ValueError("BDR axial RoPE head dimension must be divisible by 4")
        if bdr.step_loss_schedule not in ("uniform", "linear", "quadratic"):
            raise ValueError("BDR step_loss_schedule must be 'uniform', 'linear', or 'quadratic'")
        if bdr.update_rule not in ("energy_prob", "energy_dist", "free_velocity", "proposal"):
            raise ValueError("BDR update_rule must be 'energy_prob', 'energy_dist', 'free_velocity', or 'proposal'")
        if bdr.draft_view not in ("auto", "logits", "probability"):
            raise ValueError("BDR draft_view must be 'auto', 'logits', or 'probability'")
        if bdr.prediction_view not in ("auto", "logits", "probability"):
            raise ValueError("BDR prediction_view must be 'auto', 'logits', or 'probability'")
        if bdr.update_rule == "energy_prob" and bdr.update_step_size <= 0.0:
            raise ValueError("BDR update_step_size must be positive")
        self.config = config
        self.runtime = runtime
        self.bdr = bdr
        self.commit_steps = int(bdr.commit_steps)
        self.total_steps = self.commit_steps
        self.h_cycles = int(bdr.h_cycles)
        self.l_cycles = int(bdr.l_cycles)
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
        self.draft_view = self._resolve_draft_view(bdr.update_rule, bdr.draft_view)
        self.prediction_view = self._resolve_prediction_view(self.draft_view, bdr.prediction_view)
        if self.draft_view != "probability":
            raise ValueError(f"BDR update_rule={bdr.update_rule!r} requires draft_view='probability'")

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
        self.token_embed = nnx.Embed(
            self.input_vocab_size,
            config.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            embedding_init=embed_init,
            rngs=rngs,
        )
        self.puzzle_embed = CastedEmbedding(
            config.num_puzzle_identifiers,
            config.d_model,
            dtype=self.dtype,
            init_std=0.0,
            rngs=rngs,
        )
        if bdr.position_encoding == "learned":
            self.row_embed = nnx.Embed(config.grid_height, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
            self.col_embed = nnx.Embed(config.grid_width, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
            self.box_embed = nnx.Embed(self.num_boxes, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        if bdr.position_encoding == "rope":
            head_dim = self.hidden_dim // bdr.num_heads
            rope_cos, rope_sin = build_2d_axial_rope(rows, cols, head_dim, bdr.rope_theta)
            self.rope_cos = nnx.data(rope_cos)
            self.rope_sin = nnx.data(rope_sin)
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
        self.draft_to_hidden = nnx.Linear(
            self.q_vocab_size,
            self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.update_readout_norm = unscaled_rms_norm(self.hidden_dim, bdr.rms_norm_eps, self.dtype, rngs)
        self.solver_blocks = nnx.List(
            [
                BeliefDynamicsBlock(
                    config,
                    self.hidden_dim,
                    bdr.num_heads,
                    bdr.mlp_ratio,
                    self.dtype,
                    rngs=rngs,
                )
                for _ in range(bdr.l_layers)
            ]
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
        self.energy_state_to_hidden = nnx.Linear(
            self.q_vocab_size,
            self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.energy_norm = unscaled_rms_norm(self.hidden_dim, bdr.rms_norm_eps, self.dtype, rngs)
        self.energy_head = nnx.Linear(
            self.hidden_dim,
            1,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.label_energy_head = nnx.Linear(
            self.hidden_dim,
            self.q_vocab_size,
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
        self.proposal_head = nnx.Linear(
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
    def _resolve_draft_view(update_rule: str, configured: str) -> str:
        if configured != "auto":
            return configured
        del update_rule
        return "probability"

    @staticmethod
    def _resolve_prediction_view(draft_view: str, configured: str) -> str:
        if configured != "auto":
            return configured
        return draft_view

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

    def _state_is_probability(self) -> bool:
        return self.draft_view == "probability"

    def outputs_are_probabilities(self) -> bool:
        return self.prediction_view == "probability"

    def _normalize_distribution(self, distribution: Array) -> Array:
        distribution = jnp.maximum(distribution.astype(jnp.float32), self.output_logit_eps)
        return distribution / jnp.sum(distribution, axis=-1, keepdims=True)

    def _state_distribution(self, state: Array) -> Array:
        if self._state_is_probability():
            return self._normalize_distribution(state)
        return self._distribution_from_logits(state)

    def _top2_margin(self, distribution: Array) -> Array:
        distribution = distribution.astype(jnp.float32)
        top1 = jnp.max(distribution, axis=-1, keepdims=True)
        top1_index = jnp.argmax(distribution, axis=-1)
        vocab_positions = jnp.arange(distribution.shape[-1], dtype=top1_index.dtype)
        without_top1 = jnp.where(
            vocab_positions == top1_index[..., None],
            jnp.asarray(-jnp.inf, dtype=jnp.float32),
            distribution,
        )
        top2 = jnp.max(without_top1, axis=-1, keepdims=True)
        return top1 - top2

    def _logits_from_distribution(self, distribution: Array) -> Array:
        return self._center_logits(jnp.log(jnp.maximum(distribution, self.output_logit_eps)))

    def _zero_z(self, tokens: Array) -> Array:
        if self._state_is_probability():
            return jnp.broadcast_to(
                jnp.full_like(tokens[..., None], 1.0 / float(self.q_vocab_size), dtype=jnp.float32),
                (*tokens.shape, self.q_vocab_size),
            )
        # Zero centered logits represent the uniform answer distribution.
        return jnp.broadcast_to(
            jnp.zeros_like(tokens[..., None], dtype=jnp.float32),
            (*tokens.shape, self.q_vocab_size),
        )

    def _z_to_output_logits(self, z_logits: Array, tokens: Array) -> Array:
        del tokens
        if self.outputs_are_probabilities():
            return self._normalize_distribution(z_logits)
        if self._state_is_probability():
            return self._logits_from_distribution(z_logits)
        return self._center_logits(z_logits)

    def initial_z(self, tokens: Array) -> Array:
        return self._zero_z(tokens)

    def _position_embeddings(self) -> Array:
        if self.bdr.position_encoding != "learned":
            return jnp.zeros((self.config.seq_len, self.config.d_model), dtype=self.dtype)
        return (
            self.row_embed(self.row_ids)
            + self.col_embed(self.col_ids)
            + self.box_embed(self.box_ids)
        )

    def _embed_tokens(self, embedding: nnx.Embed, token_ids: Array) -> Array:
        return maybe_cast(gather_embedding_rows(embedding.embedding, token_ids.astype(jnp.int32)), self.dtype)

    def _draft_features(
        self,
        tokens: Array,
        z_logits: Array,
        step_index: Array,
    ) -> Array:
        del tokens, step_index
        return z_logits.astype(jnp.float32)

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
        draft_features = self._draft_features(
            tokens,
            z_logits,
            step_index,
        )
        context_input = self.dropout(base_embeddings, deterministic=not train, rngs=dropout_key)
        draft_input = self.dropout(draft_features, deterministic=not train, rngs=dropout_key)
        context_condition = self.context_to_hidden(maybe_cast(context_input, self.dtype)).astype(self.dtype)
        draft_condition = self.draft_to_hidden(maybe_cast(draft_input, self.dtype)).astype(self.dtype)
        return context_condition, draft_condition

    def _z_to_token_logits(self, z: Array, tokens: Array, step_index: Array) -> Array:
        del step_index
        return self._z_to_output_logits(z, tokens)

    def _energy_per_cell_from_read_state(
        self,
        candidate_distribution: Array,
        read_state: Array,
    ) -> Array:
        energy_condition = self.energy_state_to_hidden(
            maybe_cast(candidate_distribution - (1.0 / float(self.q_vocab_size)), self.dtype)
        ).astype(jnp.float32)
        energy_input = self.energy_norm(
            (read_state.astype(jnp.float32) + energy_condition).astype(self.dtype)
        )
        return self.energy_head(maybe_cast(energy_input, self.dtype)).astype(jnp.float32)[..., 0]

    def _label_energies_from_read_state(self, read_state: Array) -> Array:
        return self.label_energy_head(maybe_cast(read_state, self.dtype)).astype(jnp.float32)

    def _distribution_energy_step_from_read_state(
        self,
        current_distribution: Array,
        read_state: Array,
    ) -> tuple[Array, dict[str, Array], tuple[Array, Array]]:
        current_distribution = self._normalize_distribution(current_distribution)
        label_energies = self._label_energies_from_read_state(read_state)
        next_distribution = jax.nn.softmax(-label_energies, axis=-1)
        distribution_step = next_distribution - current_distribution
        distribution_tv_delta = 0.5 * jnp.sum(
            jnp.abs(distribution_step),
            axis=-1,
            keepdims=True,
        )
        energy_update_rms = jnp.sqrt(jnp.mean(jnp.square(distribution_step), axis=-1, keepdims=True) + 1e-12)
        energy_value = jnp.sum(current_distribution * label_energies, axis=-1, keepdims=True)
        energy_entropy = -jnp.sum(
            next_distribution * jnp.log(jnp.maximum(next_distribution, self.output_logit_eps)),
            axis=-1,
            keepdims=True,
        ) / jnp.log(float(self.q_vocab_size))
        path_energy = jnp.mean(jnp.square(distribution_step), axis=-1, keepdims=True)
        zero = jnp.zeros_like(energy_update_rms)
        diagnostics = {
            "update_step_size": zero,
            "update_rms": energy_update_rms,
            "velocity_rms": zero,
            "energy_update_rms": energy_update_rms,
            "energy_value": energy_value,
            "energy_grad_rms": zero,
            "logit_step_rms": zero,
            "energy_distribution_step_rms": energy_update_rms,
            "energy_entropy": energy_entropy,
            "proposal_update_rms": zero,
            "proposal_entropy": zero,
            "free_velocity_rms": zero,
            "free_velocity_negative_rate": zero,
            "distribution_tv_delta": distribution_tv_delta,
            "path_energy": path_energy,
        }
        return next_distribution, diagnostics, (current_distribution, next_distribution)

    def _energy_probability_descent_from_read_state(
        self,
        current_distribution: Array,
        read_state: Array,
    ) -> tuple[Array, dict[str, Array], tuple[Array, Array]]:
        current_distribution = self._normalize_distribution(current_distribution)

        def energy_per_cell(candidate_distribution: Array) -> Array:
            candidate_distribution = self._normalize_distribution(candidate_distribution)
            return self._energy_per_cell_from_read_state(candidate_distribution, read_state)

        energy_cells, energy_vjp = jax.vjp(energy_per_cell, current_distribution)
        energy_value = energy_cells[..., None]
        energy_grad = energy_vjp(jnp.ones_like(energy_cells))[0].astype(jnp.float32)
        energy_grad = energy_grad - jnp.mean(energy_grad, axis=-1, keepdims=True)
        update_step_size_scalar = float(self.bdr.update_step_size)
        distribution_step = -update_step_size_scalar * energy_grad
        next_distribution = self._normalize_distribution(current_distribution + distribution_step)

        distribution_tv_delta = 0.5 * jnp.sum(
            jnp.abs(next_distribution - current_distribution),
            axis=-1,
            keepdims=True,
        )
        energy_grad_rms = jnp.sqrt(jnp.mean(jnp.square(energy_grad), axis=-1, keepdims=True) + 1e-12)
        distribution_step_rms = jnp.sqrt(jnp.mean(jnp.square(distribution_step), axis=-1, keepdims=True) + 1e-12)
        update_step_size = jnp.broadcast_to(
            jnp.asarray(update_step_size_scalar, dtype=jnp.float32),
            (*current_distribution.shape[:-1], 1),
        )
        path_energy = jnp.mean(jnp.square(next_distribution - current_distribution), axis=-1, keepdims=True)
        diagnostics = {
            "update_step_size": update_step_size,
            "update_rms": energy_grad_rms,
            "velocity_rms": jnp.zeros_like(energy_grad_rms),
            "energy_update_rms": energy_grad_rms,
            "energy_value": energy_value,
            "energy_grad_rms": energy_grad_rms,
            "logit_step_rms": distribution_step_rms,
            "energy_distribution_step_rms": jnp.zeros_like(energy_grad_rms),
            "energy_entropy": jnp.zeros_like(energy_grad_rms),
            "proposal_update_rms": jnp.zeros_like(energy_grad_rms),
            "proposal_entropy": jnp.zeros_like(energy_grad_rms),
            "free_velocity_rms": jnp.zeros_like(energy_grad_rms),
            "free_velocity_negative_rate": jnp.zeros_like(energy_grad_rms),
            "distribution_tv_delta": distribution_tv_delta,
            "path_energy": path_energy,
        }
        return next_distribution, diagnostics, (current_distribution, next_distribution)

    def _proposal_step_from_read_state(
        self,
        current_distribution: Array,
        read_state: Array,
    ) -> tuple[Array, dict[str, Array], tuple[Array, Array]]:
        current_distribution = self._normalize_distribution(current_distribution)
        proposal_logits = self.proposal_head(maybe_cast(read_state, self.dtype)).astype(jnp.float32)
        next_distribution = jax.nn.softmax(proposal_logits, axis=-1)
        distribution_step = next_distribution - current_distribution
        distribution_tv_delta = 0.5 * jnp.sum(
            jnp.abs(distribution_step),
            axis=-1,
            keepdims=True,
        )
        proposal_update_rms = jnp.sqrt(jnp.mean(jnp.square(distribution_step), axis=-1, keepdims=True) + 1e-12)
        proposal_entropy = -jnp.sum(
            next_distribution * jnp.log(jnp.maximum(next_distribution, self.output_logit_eps)),
            axis=-1,
            keepdims=True,
        ) / jnp.log(float(self.q_vocab_size))
        path_energy = jnp.mean(jnp.square(distribution_step), axis=-1, keepdims=True)
        zero = jnp.zeros_like(proposal_update_rms)
        diagnostics = {
            "update_step_size": zero,
            "update_rms": proposal_update_rms,
            "velocity_rms": zero,
            "energy_update_rms": zero,
            "energy_value": zero,
            "energy_grad_rms": zero,
            "logit_step_rms": zero,
            "energy_distribution_step_rms": zero,
            "energy_entropy": zero,
            "proposal_update_rms": proposal_update_rms,
            "proposal_entropy": proposal_entropy,
            "free_velocity_rms": zero,
            "free_velocity_negative_rate": zero,
            "distribution_tv_delta": distribution_tv_delta,
            "path_energy": path_energy,
        }
        return next_distribution, diagnostics, (current_distribution, next_distribution)

    def _free_velocity_step_from_read_state(
        self,
        current_distribution: Array,
        read_state: Array,
    ) -> tuple[Array, dict[str, Array], tuple[Array, Array]]:
        current_distribution = self._normalize_distribution(current_distribution)
        raw_delta = self.velocity_head(maybe_cast(read_state, self.dtype)).astype(jnp.float32)
        delta = raw_delta - jnp.mean(raw_delta, axis=-1, keepdims=True)
        pre_normalized = current_distribution + delta
        negative_rate = jnp.mean((pre_normalized < 0.0).astype(jnp.float32), axis=-1, keepdims=True)
        next_distribution = self._normalize_distribution(jax.nn.relu(pre_normalized) + self.output_logit_eps)
        distribution_step = next_distribution - current_distribution
        distribution_tv_delta = 0.5 * jnp.sum(
            jnp.abs(distribution_step),
            axis=-1,
            keepdims=True,
        )
        free_velocity_rms = jnp.sqrt(jnp.mean(jnp.square(delta), axis=-1, keepdims=True) + 1e-12)
        path_energy = jnp.mean(jnp.square(distribution_step), axis=-1, keepdims=True)
        zero = jnp.zeros_like(free_velocity_rms)
        diagnostics = {
            "update_step_size": zero,
            "update_rms": free_velocity_rms,
            "velocity_rms": zero,
            "energy_update_rms": zero,
            "energy_value": zero,
            "energy_grad_rms": zero,
            "logit_step_rms": zero,
            "energy_distribution_step_rms": zero,
            "energy_entropy": zero,
            "proposal_update_rms": zero,
            "proposal_entropy": zero,
            "free_velocity_rms": free_velocity_rms,
            "free_velocity_negative_rate": negative_rate,
            "distribution_tv_delta": distribution_tv_delta,
            "path_energy": path_energy,
        }
        return next_distribution, diagnostics, (current_distribution, next_distribution)

    def _hidden_h_cycle(
        self,
        hidden_state: Array,
        context_condition: Array,
        draft_condition: Array,
    ) -> Array:
        hidden = hidden_state.astype(self.dtype)

        def latent_update(state: Array) -> Array:
            def refine_body(_step: Array, carry: Array) -> Array:
                del _step
                refined = carry
                for block in self.solver_blocks:
                    refined = block.global_communicate(
                        refined,
                        context_condition,
                        draft_condition,
                        self.rope_cos,
                        self.rope_sin,
                    )
                    refined = block.local_think(refined, context_condition, draft_condition)
                return refined

            return jax.lax.fori_loop(0, self.l_cycles, refine_body, state)

        return run_truncated_h_cycles(
            hidden,
            h_cycles=self.h_cycles,
            latent_update=latent_update,
        )

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
    ) -> tuple[Array, Array, dict[str, Array], tuple[Array, Array]]:
        # The explicit z state is stopped before context construction; the
        # configured update rule then acts on logits using the refined H.
        context_condition, draft_condition = self._typed_conditions(
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
            draft_condition,
        )
        read_state = self._update_read_state(hidden_state)
        del step_index
        if self.bdr.update_rule == "energy_dist":
            z, update_diagnostics, update_distributions = self._distribution_energy_step_from_read_state(z, read_state)
        elif self.bdr.update_rule == "free_velocity":
            z, update_diagnostics, update_distributions = self._free_velocity_step_from_read_state(z, read_state)
        elif self.bdr.update_rule == "proposal":
            z, update_diagnostics, update_distributions = self._proposal_step_from_read_state(z, read_state)
        elif self.bdr.update_rule == "energy_prob":
            z, update_diagnostics, update_distributions = self._energy_probability_descent_from_read_state(z, read_state)
        else:
            raise ValueError(f"Unsupported BDR update_rule={self.bdr.update_rule!r}")
        update_diagnostics = {
            **update_diagnostics,
            "halt_logits": self._halt_logits(read_state, tokens),
        }
        return (
            z,
            hidden_state,
            update_diagnostics,
            update_distributions,
        )

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
        return jnp.broadcast_to(
            jnp.zeros_like(tokens[..., None], dtype=self.dtype),
            (tokens.shape[0], self.config.seq_len, self.hidden_dim),
        )

    def context_memory(
        self,
        tokens: Array,
        puzzle_identifiers: Array | None = None,
        puzzle_embeddings: Array | None = None,
    ) -> tuple[Array, Array]:
        context = self.context_mask(tokens)
        token_ids = jnp.clip(tokens, 0, self.input_vocab_size - 1)
        base_embeddings = self._embed_tokens(self.token_embed, token_ids) * self.embed_scale
        if puzzle_embeddings is None:
            if puzzle_identifiers is None:
                puzzle_identifiers = jnp.zeros((tokens.shape[0],), dtype=jnp.int32)
            puzzle_embedding = self.puzzle_embed(
                jnp.clip(puzzle_identifiers.astype(jnp.int32), 0, self.config.num_puzzle_identifiers - 1),
                train=True,
            )
        else:
            puzzle_embedding = puzzle_embeddings
        base_embeddings = base_embeddings + puzzle_embedding[:, None, :]
        if self.bdr.position_encoding == "learned":
            base_embeddings = base_embeddings + self._position_embeddings()[None, :, :]
        return base_embeddings, context

    def initial_carry(self, batch: dict[str, Array]) -> dict[str, Array]:
        batch_size = batch["inputs"].shape[0]
        if "puzzle_identifiers" in batch:
            puzzle_identifiers = batch["puzzle_identifiers"].astype(jnp.int32)
        else:
            puzzle_identifiers = jnp.zeros((batch_size,), dtype=jnp.int32)
        return {
            "z": self._zero_z(batch["inputs"]),
            "hidden": jnp.zeros(
                (batch_size, self.config.seq_len, self.hidden_dim),
                dtype=self.dtype,
            ),
            "steps": jnp.zeros((batch_size,), dtype=jnp.int32),
            "halted": jnp.ones((batch_size,), dtype=bool),
            "current_inputs": jnp.zeros_like(batch["inputs"]),
            "current_labels": jnp.zeros_like(batch["labels"]),
            "current_puzzle_identifiers": jnp.zeros_like(puzzle_identifiers),
        }

    def _halt_logits(self, read_state: Array, tokens: Array) -> Array:
        query_mask = (~self.context_mask(tokens)).astype(jnp.float32)
        all_cells_mask = jnp.ones_like(query_mask)
        has_query = jnp.sum(query_mask, axis=-1, keepdims=True) > 0.0
        mask = jnp.where(has_query, query_mask, all_cells_mask)
        pooled = (
            jnp.sum(read_state.astype(jnp.float32) * mask[..., None], axis=1)
            / jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1.0)
        )
        return self.halt_head(maybe_cast(pooled, self.dtype)).astype(jnp.float32)[..., 0]

    def forward_act_step(
        self,
        carry: dict[str, Array],
        batch: dict[str, Array],
        *,
        train: bool,
        dropout_key: Array | None = None,
        puzzle_embeddings: Array | None = None,
    ) -> tuple[dict[str, Array], Array, dict[str, Array]]:
        if dropout_key is None:
            dropout_key = jax.random.key(0)
        reset = carry["halted"]
        reset_cells = reset[:, None]
        reset_state = reset[:, None, None]
        inputs = jnp.where(reset_cells, batch["inputs"], carry["current_inputs"])
        labels = jnp.where(reset_cells, batch["labels"], carry["current_labels"])
        if "puzzle_identifiers" in batch:
            batch_puzzle_identifiers = batch["puzzle_identifiers"].astype(jnp.int32)
        else:
            batch_puzzle_identifiers = jnp.zeros((inputs.shape[0],), dtype=jnp.int32)
        puzzle_identifiers = jnp.where(
            reset,
            batch_puzzle_identifiers,
            carry["current_puzzle_identifiers"],
        )
        steps = jnp.where(reset, 0, carry["steps"])

        base_embeddings, _context = self.context_memory(inputs, puzzle_identifiers, puzzle_embeddings)
        step_index = jnp.minimum(steps, self.total_steps - 1)
        reset_z = self.initial_z(inputs)
        z = jnp.where(reset_state, reset_z, carry["z"])
        reset_hidden = self.initial_hidden_state(
            inputs,
            reset_z,
            base_embeddings,
            train=train,
            dropout_key=dropout_key,
        )
        hidden = jnp.where(reset_state, reset_hidden, carry["hidden"])
        next_z, next_hidden, update_diagnostics, update_distributions = self._commit_step(
            inputs,
            z,
            hidden,
            base_embeddings,
            step_index,
            train=train,
            dropout_key=dropout_key,
            stop_hidden_between_steps=True,
        )
        halt_logits = update_diagnostics["halt_logits"]
        if self._state_is_probability():
            _current_distribution, next_distribution = update_distributions
            logits = next_distribution
        else:
            logits = self._z_to_token_logits(next_z, inputs, step_index)
        new_steps = steps + 1
        is_last_step = new_steps >= self.total_steps
        if train:
            next_halted = is_last_step | (halt_logits > 0.0)
            if self.total_steps > 1 and self.bdr.halt_exploration_prob > 0.0:
                explore_key, min_step_key = jax.random.split(dropout_key)
                use_random_min = jax.random.uniform(explore_key, (inputs.shape[0],)) < self.bdr.halt_exploration_prob
                random_min_steps = jax.random.randint(
                    min_step_key,
                    (inputs.shape[0],),
                    minval=2,
                    maxval=self.total_steps + 1,
                )
                min_steps = jnp.where(use_random_min, random_min_steps, 1)
                next_halted = next_halted & (new_steps >= min_steps)
        else:
            next_halted = is_last_step
        new_carry = {
            "z": jax.lax.stop_gradient(next_z),
            "hidden": jax.lax.stop_gradient(next_hidden),
            "steps": jax.lax.stop_gradient(new_steps),
            "halted": jax.lax.stop_gradient(next_halted),
            "current_inputs": inputs,
            "current_labels": labels,
            "current_puzzle_identifiers": puzzle_identifiers,
        }
        diagnostics = {
            "halt_logits": halt_logits,
            "act_step": jnp.mean(new_steps.astype(jnp.float32)),
            "halted_rate": jnp.mean(next_halted.astype(jnp.float32)),
            "reset_rate": jnp.mean(reset.astype(jnp.float32)),
            "update_step_size": jnp.mean(update_diagnostics["update_step_size"].astype(jnp.float32)),
            "update_rms": jnp.mean(update_diagnostics["update_rms"].astype(jnp.float32)),
            "velocity_rms": jnp.mean(update_diagnostics["velocity_rms"].astype(jnp.float32)),
            "energy_update_rms": jnp.mean(update_diagnostics["energy_update_rms"].astype(jnp.float32)),
            "energy_value": jnp.mean(update_diagnostics["energy_value"].astype(jnp.float32)),
            "energy_grad_rms": jnp.mean(update_diagnostics["energy_grad_rms"].astype(jnp.float32)),
            "logit_step_rms": jnp.mean(update_diagnostics["logit_step_rms"].astype(jnp.float32)),
            "energy_distribution_step_rms": jnp.mean(update_diagnostics["energy_distribution_step_rms"].astype(jnp.float32)),
            "energy_entropy": jnp.mean(update_diagnostics["energy_entropy"].astype(jnp.float32)),
            "proposal_update_rms": jnp.mean(update_diagnostics["proposal_update_rms"].astype(jnp.float32)),
            "proposal_entropy": jnp.mean(update_diagnostics["proposal_entropy"].astype(jnp.float32)),
            "free_velocity_rms": jnp.mean(update_diagnostics["free_velocity_rms"].astype(jnp.float32)),
            "free_velocity_negative_rate": jnp.mean(update_diagnostics["free_velocity_negative_rate"].astype(jnp.float32)),
            "distribution_tv_delta": jnp.mean(update_diagnostics["distribution_tv_delta"].astype(jnp.float32)),
            "path_energy": jnp.mean(update_diagnostics["path_energy"].astype(jnp.float32)),
        }
        return new_carry, logits, diagnostics

    def run_commit_steps(
        self,
        tokens: Array,
        *,
        initial_z: Array | None = None,
        puzzle_identifiers: Array | None = None,
        train: bool,
        dropout_key: Array | None = None,
        return_final_only: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        if initial_z is None:
            initial_z = self.initial_z(tokens)
        base_embeddings, context = self.context_memory(tokens, puzzle_identifiers)
        query_mask = (~context).astype(jnp.float32)
        query_normalizer = jnp.maximum(jnp.sum(query_mask), 1.0)

        def scan_step(carry, scan_inputs):
            step_index, step_dropout_key = scan_inputs
            z, hidden_state = carry
            prev_hidden_state = hidden_state
            next_z, next_hidden, update_diagnostics, _update_distributions = self._commit_step(
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
            if self._state_is_probability():
                _current_distribution, next_distribution = _update_distributions
                logits = next_distribution
                confidence = jnp.max(next_distribution, axis=-1)
            else:
                logits = self._z_to_token_logits(next_z, tokens, step_index)
                confidence = jnp.max(self._state_distribution(next_z), axis=-1)
            q_top1_probability = jnp.sum(confidence * query_mask) / query_normalizer
            hidden_delta = jnp.mean(
                jnp.linalg.norm((next_hidden - prev_hidden_state).astype(jnp.float32), axis=-1)
            )
            return next_carry, (
                logits,
                update_diagnostics["halt_logits"],
                hidden_delta,
                q_top1_probability,
                jnp.mean(update_diagnostics["update_step_size"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["update_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["velocity_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["energy_update_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["energy_value"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["energy_grad_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["logit_step_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["energy_distribution_step_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["energy_entropy"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["proposal_update_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["proposal_entropy"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["free_velocity_rms"].astype(jnp.float32)),
                jnp.mean(update_diagnostics["free_velocity_negative_rate"].astype(jnp.float32)),
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
                "halt_logits": jnp.zeros((self.total_steps, tokens.shape[0]), dtype=jnp.float32),
                "hidden_delta_mean": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "unroll_steps": jnp.asarray(self.total_steps, dtype=jnp.float32),
                "update_step_size": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "update_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "velocity_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "energy_update_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "energy_value": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "energy_grad_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "logit_step_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "energy_distribution_step_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "energy_entropy": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "proposal_update_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "proposal_entropy": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "free_velocity_rms": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "free_velocity_negative_rate": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "distribution_tv_delta": jnp.zeros((self.total_steps,), dtype=jnp.float32),
                "path_energy": jnp.zeros((self.total_steps,), dtype=jnp.float32),
            }
            return logits, diagnostics
        (
            step_logits,
            halt_logits,
            hidden_delta_mean,
            q_top1_probability,
            update_step_size,
            update_rms,
            velocity_rms,
            energy_update_rms,
            energy_value,
            energy_grad_rms,
            logit_step_rms,
            energy_distribution_step_rms,
            energy_entropy,
            proposal_update_rms,
            proposal_entropy,
            free_velocity_rms,
            free_velocity_negative_rate,
            distribution_tv_delta,
            path_energy,
        ) = scan_outputs
        diagnostics = {
            "q_top1_probability": q_top1_probability,
            "halt_logits": halt_logits,
            "hidden_delta_mean": hidden_delta_mean,
            "unroll_steps": jnp.asarray(self.total_steps, dtype=jnp.float32),
            "update_step_size": update_step_size,
            "update_rms": update_rms,
            "velocity_rms": velocity_rms,
            "energy_update_rms": energy_update_rms,
            "energy_value": energy_value,
            "energy_grad_rms": energy_grad_rms,
            "logit_step_rms": logit_step_rms,
            "energy_distribution_step_rms": energy_distribution_step_rms,
            "energy_entropy": energy_entropy,
            "proposal_update_rms": proposal_update_rms,
            "proposal_entropy": proposal_entropy,
            "free_velocity_rms": free_velocity_rms,
            "free_velocity_negative_rate": free_velocity_negative_rate,
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
        puzzle_identifiers: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        return self.run_commit_steps(
            tokens,
            initial_z=initial_z,
            puzzle_identifiers=puzzle_identifiers,
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
        puzzle_identifiers: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        return self.forward_all_steps_with_diagnostics(
            tokens,
            train=train,
            dropout_key=dropout_key,
            initial_z=initial_z,
            puzzle_identifiers=puzzle_identifiers,
        )

    def forward_all_steps(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        puzzle_identifiers: Array | None = None,
    ) -> Array:
        logits, _ = self.forward_all_steps_with_diagnostics(
            tokens,
            train=train,
            dropout_key=dropout_key,
            puzzle_identifiers=puzzle_identifiers,
        )
        return logits

    def forward_final(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        puzzle_identifiers: Array | None = None,
    ) -> Array:
        logits, _ = self.forward_final_with_diagnostics(
            tokens,
            train=train,
            dropout_key=dropout_key,
            puzzle_identifiers=puzzle_identifiers,
        )
        return logits[-1]

    def __call__(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        puzzle_identifiers: Array | None = None,
    ) -> Array:
        return self.forward_final(
            tokens,
            train=train,
            dropout_key=dropout_key,
            puzzle_identifiers=puzzle_identifiers,
        )
