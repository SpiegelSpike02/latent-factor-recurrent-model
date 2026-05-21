from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from lfrm.config import ModelConfig, RuntimeConfig
from .common import Array, casted_linear_init, compute_dtype, maybe_cast, trunc_normal_init
from .recurrent.layers import FullAttention, rms_norm as _shared_rms_norm


def _rms_norm(x: Array, eps: float = 1e-5) -> Array:
    return _shared_rms_norm(x, eps)


class BRCSolverBlock(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        num_heads: int,
        mlp_ratio: int,
        dtype: jnp.dtype,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.dtype = dtype
        d_model = config.d_model
        hidden_dim = max(d_model, mlp_ratio * d_model)
        self.attention = FullAttention(
            d_model,
            num_heads,
            dtype,
            name="BRC",
            rngs=rngs,
        )
        self.msg_in = nnx.Linear(
            d_model,
            hidden_dim,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.msg_out = nnx.Linear(
            hidden_dim,
            d_model,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
    def __call__(
        self,
        h: Array,
        rope_cos: Array | None,
        rope_sin: Array | None,
    ) -> Array:
        attn = self.attention(h, rope_cos=rope_cos, rope_sin=rope_sin)
        h = _rms_norm(h.astype(jnp.float32) + attn.astype(jnp.float32), self.config.brc_config.rms_norm_eps).astype(self.dtype)
        msg = self.msg_out(jax.nn.silu(self.msg_in(maybe_cast(h, self.dtype)))).astype(jnp.float32)
        return _rms_norm(h.astype(jnp.float32) + msg, self.config.brc_config.rms_norm_eps).astype(self.dtype)


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
        if brc.recurrent_steps < 1:
            raise ValueError("BRC recurrent_steps must be at least 1")
        if brc.block_layers < 1:
            raise ValueError("BRC block_layers must be at least 1")
        if brc.num_heads < 1:
            raise ValueError("BRC num_heads must be at least 1")
        if config.d_model % brc.num_heads != 0:
            raise ValueError("BRC d_model must be divisible by num_heads")
        if brc.mlp_ratio < 1:
            raise ValueError("BRC mlp_ratio must be at least 1")
        if brc.position_encoding not in ("rope", "learned", "none"):
            raise ValueError("BRC position_encoding must be 'rope', 'learned', or 'none'")
        if brc.position_encoding == "rope" and (config.d_model // brc.num_heads) % 2 != 0:
            raise ValueError("BRC RoPE head dimension must be even")
        if brc.step_loss_schedule not in ("uniform", "linear"):
            raise ValueError("BRC step_loss_schedule must be 'uniform' or 'linear'")
        if not 0.0 <= brc.denoise_initial_prob <= 1.0:
            raise ValueError("BRC denoise_initial_prob must be in [0, 1]")
        if not 0.0 <= brc.denoise_trajectory_prob <= 1.0:
            raise ValueError("BRC denoise_trajectory_prob must be in [0, 1]")
        if not 0.0 <= brc.denoise_teacher_reveal_prob <= 1.0:
            raise ValueError("BRC denoise_teacher_reveal_prob must be in [0, 1]")
        if len(brc.denoise_mode_weights) != 3:
            raise ValueError("BRC denoise_mode_weights must contain three weights")
        if any(weight < 0.0 for weight in brc.denoise_mode_weights):
            raise ValueError("BRC denoise_mode_weights must be non-negative")
        if sum(brc.denoise_mode_weights) <= 0.0:
            raise ValueError("BRC denoise_mode_weights must contain a positive weight")
        if brc.fixed_point_entropy_weight < 0.0:
            raise ValueError("BRC fixed_point_entropy_weight must be non-negative")
        if min(brc.fixed_point_loss_weight, brc.context_weight_reg_weight) < 0.0:
            raise ValueError("BRC fixed-point loss weights must be non-negative")

        self.config = config
        self.runtime = runtime
        self.brc = brc
        self.recurrent_steps = int(brc.recurrent_steps)
        self.dtype = compute_dtype(runtime.compute_dtype)
        self.embed_scale = math.sqrt(config.d_model)
        self.belief_vocab_size = config.vocab_size
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
        self.draft_embed = nnx.Embed(
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
        self.time_embed = nnx.Embed(self.recurrent_steps, config.d_model, dtype=self.dtype, param_dtype=jnp.float32, embedding_init=embed_init, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        if brc.position_encoding == "rope":
            head_dim = config.d_model // brc.num_heads
            inv_freq = 1.0 / (brc.rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
            positions = jnp.arange(config.seq_len, dtype=jnp.float32)
            freqs = positions[:, None] * inv_freq[None, :]
            rope = jnp.concatenate((freqs, freqs), axis=-1)
            self.rope_cos = nnx.data(jnp.cos(rope))
            self.rope_sin = nnx.data(jnp.sin(rope))
        else:
            self.rope_cos = None
            self.rope_sin = None

        self.input_to_scratch = nnx.Linear(
            config.d_model,
            config.d_model,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.solver_blocks = nnx.List(
            [
                BRCSolverBlock(
                    config,
                    brc.num_heads,
                    brc.mlp_ratio,
                    self.dtype,
                    rngs=rngs,
                )
                for _ in range(brc.block_layers)
            ]
        )
        self.lm_head = nnx.Linear(
            config.d_model,
            config.vocab_size,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.step_gate = nnx.Linear(
            config.d_model,
            1,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.energy_context_gate = nnx.Linear(
            config.d_model,
            1,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
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

    def draft_from_tokens(self, tokens: Array) -> Array:
        return tokens.astype(jnp.int32)

    def initial_draft(self, tokens: Array) -> Array:
        return self.draft_from_tokens(tokens)

    def _normalize_belief_logits(self, belief_logits: Array, tokens: Array) -> Array:
        del tokens
        return belief_logits.astype(jnp.float32)

    def initial_belief_logits(self, tokens: Array) -> Array:
        zeros = jnp.zeros((*tokens.shape, self.belief_vocab_size), dtype=jnp.float32)
        return self._normalize_belief_logits(zeros, tokens)

    def belief_logits_from_tokens(self, puzzle: Array, candidate: Array, mask: Array | None = None) -> Array:
        class_ids = jnp.clip(candidate, 0, self.config.vocab_size - 1)
        hard_logits = -1.0e4 + 2.0e4 * jax.nn.one_hot(class_ids.astype(jnp.int32), self.belief_vocab_size)
        if mask is not None:
            hard_logits = jnp.where(mask[..., None], hard_logits, 0.0)
        return self._normalize_belief_logits(hard_logits, puzzle)

    def belief_logits_from_draft(self, tokens: Array, draft: Array) -> Array:
        class_ids = jnp.clip(draft, 0, self.config.vocab_size - 1)
        active = jnp.ones_like(draft, dtype=bool)
        hard_logits = -1.0e4 + 2.0e4 * jax.nn.one_hot(class_ids.astype(jnp.int32), self.belief_vocab_size)
        hard_logits = jnp.where(active[..., None], hard_logits, 0.0)
        return self._normalize_belief_logits(hard_logits, tokens)

    def _position_embeddings(self) -> Array:
        if self.brc.position_encoding != "learned":
            return jnp.zeros((self.config.seq_len, self.config.d_model), dtype=self.dtype)
        return (
            self.row_embed(self.row_ids)
            + self.col_embed(self.col_ids)
            + self.box_embed(self.box_ids)
        )

    def _belief_embedding(self, tokens: Array, belief_logits: Array) -> Array:
        belief_probs = jax.nn.softmax(self._normalize_belief_logits(belief_logits, tokens), axis=-1)
        embedding_table = maybe_cast(self.draft_embed.embedding[: self.config.vocab_size], self.dtype)
        return jnp.einsum(
            "bnd,dk->bnk",
            maybe_cast(belief_probs, self.dtype),
            embedding_table,
            preferred_element_type=jnp.float32,
        )

    def _cell_embeddings(
        self,
        tokens: Array,
        belief_logits: Array,
        base_embeddings: Array,
        time_embedding: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> Array:
        belief_embedding = self._belief_embedding(tokens, belief_logits)
        x = (
            base_embeddings
            + belief_embedding
            + time_embedding[None, None, :]
        ) * math.sqrt(1.0 / 5.0)
        return self.dropout(x, deterministic=not train, rngs=dropout_key).astype(self.dtype)

    def _belief_to_token_logits(self, belief_logits: Array, tokens: Array, step_index: Array) -> Array:
        del step_index
        return self._normalize_belief_logits(belief_logits, tokens)

    def _belief_update(self, tokens: Array, belief_logits: Array, raw_delta: Array, alpha: Array, step_index: Array) -> Array:
        del step_index
        delta = raw_delta.astype(jnp.float32)
        next_belief = belief_logits.astype(jnp.float32) + alpha.astype(jnp.float32) * delta
        return self._normalize_belief_logits(next_belief, tokens)

    def _fixed_point_stats(
        self,
        belief_logits: Array,
        next_belief: Array,
        context_features: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        prev_log_probs = jax.nn.log_softmax(belief_logits.astype(jnp.float32), axis=-1)
        next_log_probs = jax.nn.log_softmax(next_belief.astype(jnp.float32), axis=-1)
        prev_probs = jax.lax.stop_gradient(jnp.exp(prev_log_probs))
        next_probs = jnp.exp(next_log_probs)
        cell_kl = jnp.sum(prev_probs * (prev_log_probs - next_log_probs), axis=-1)
        cell_entropy = -jnp.sum(next_probs * next_log_probs, axis=-1)
        kl_delta = jnp.mean(cell_kl)
        entropy = jnp.mean(cell_entropy)
        confidence = jnp.mean(jnp.max(next_probs, axis=-1))
        cell_update_norm = jnp.linalg.norm((next_belief - belief_logits).astype(jnp.float32), axis=-1)
        update_norm = jnp.mean(cell_update_norm)
        context_weight = 2.0 * jax.nn.sigmoid(
            self.energy_context_gate(maybe_cast(context_features, self.dtype)).astype(jnp.float32)
        ).squeeze(-1)
        cell_energy = cell_update_norm + self.brc.fixed_point_entropy_weight * cell_entropy
        energy = jnp.sum(context_weight * cell_energy) / jnp.maximum(jnp.sum(context_weight), 1.0)
        context_weight_reg = jnp.mean(jnp.square(context_weight - 1.0))
        context_weight_mean = jnp.mean(context_weight)
        return energy, kl_delta, entropy, confidence, update_norm, context_weight_reg, context_weight_mean

    def _scratch_refine(self, cell_input: Array) -> tuple[Array, Array, dict[str, Array]]:
        scratch = self.input_to_scratch(maybe_cast(cell_input, self.dtype)).astype(self.dtype)
        for block in self.solver_blocks:
            scratch = block(scratch, self.rope_cos, self.rope_sin)
        alpha = jax.nn.sigmoid(self.step_gate(maybe_cast(scratch, self.dtype)).astype(jnp.float32))
        return scratch, alpha, {
            "step_gate_mean": jnp.mean(alpha),
            "step_gate_std": jnp.std(alpha),
        }

    def context_memory(
        self,
        tokens: Array,
    ) -> tuple[Array, Array]:
        position_embeddings = self._position_embeddings()
        context = self.context_mask(tokens)
        base_embeddings = (
            self.puzzle_embed(tokens.astype(jnp.int32))
            + self.context_embed(context.astype(jnp.int32))
            + position_embeddings[None, :, :]
        )
        return base_embeddings, context

    def run_diffusion(
        self,
        tokens: Array,
        *,
        initial_draft: Array | None = None,
        initial_belief: Array | None = None,
        train: bool,
        dropout_key: Array | None = None,
        return_raw_final_logits: bool = False,
        return_final_only: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        if initial_belief is None:
            if initial_draft is None:
                initial_belief = self.initial_belief_logits(tokens)
            else:
                initial_belief = self.belief_logits_from_draft(tokens, initial_draft)
        base_embeddings, context = self.context_memory(tokens)
        query_mask = (~context).astype(jnp.float32)
        query_normalizer = jnp.maximum(jnp.sum(query_mask), 1.0)

        def scan_step(carry, scan_inputs):
            step_index, step_dropout_key, time_embedding = scan_inputs
            if return_raw_final_logits:
                belief_logits, _raw_final_logits = carry
            else:
                belief_logits = carry
            cell_input = self._cell_embeddings(
                tokens,
                belief_logits,
                base_embeddings,
                time_embedding,
                train=train,
                dropout_key=step_dropout_key,
            )
            scratch, alpha, block_diagnostics = self._scratch_refine(cell_input)
            raw_delta = self.lm_head(maybe_cast(scratch, self.dtype))
            next_belief = self._belief_update(tokens, belief_logits, raw_delta, alpha, step_index)
            next_carry = next_belief
            if return_raw_final_logits:
                next_carry = (next_belief, raw_delta.astype(jnp.float32))
            if return_final_only:
                return next_carry, None
            logits = self._belief_to_token_logits(next_belief, tokens, step_index)
            belief_probs = jax.nn.softmax(next_belief, axis=-1)
            confidence = jnp.max(belief_probs, axis=-1)
            filled_ratio = jnp.sum(confidence * query_mask) / query_normalizer
            (
                energy,
                kl_delta,
                entropy,
                mean_confidence,
                update_norm,
                context_weight_reg,
                context_weight_mean,
            ) = self._fixed_point_stats(
                belief_logits,
                next_belief,
                base_embeddings,
            )
            return next_carry, (
                logits,
                filled_ratio,
                block_diagnostics["step_gate_mean"],
                block_diagnostics["step_gate_std"],
                energy,
                kl_delta,
                entropy,
                mean_confidence,
                update_norm,
                context_weight_reg,
                context_weight_mean,
            )

        step_indices = jnp.arange(self.recurrent_steps, dtype=jnp.int32)
        if dropout_key is None:
            step_dropout_keys = jax.random.split(jax.random.key(0), self.recurrent_steps)
        else:
            step_dropout_keys = jax.random.split(dropout_key, self.recurrent_steps)
        time_embeddings = self.time_embed(step_indices)
        initial_carry = initial_belief.astype(jnp.float32)
        if return_raw_final_logits:
            raw_final0 = jnp.zeros((*tokens.shape, self.config.vocab_size), dtype=jnp.float32)
            initial_carry = (initial_belief.astype(jnp.float32), raw_final0)
        final_carry, scan_outputs = jax.lax.scan(
            scan_step,
            initial_carry,
            (step_indices, step_dropout_keys, time_embeddings),
        )
        if return_raw_final_logits:
            belief_final, raw_final_logits = final_carry
        else:
            belief_final = final_carry
        (
            final_energy,
            final_kl_delta,
            final_entropy,
            final_confidence,
            final_update_norm,
            final_context_weight_reg,
            final_context_weight_mean,
        ) = self._fixed_point_stats(
            initial_belief.astype(jnp.float32),
            belief_final,
            base_embeddings,
        )
        if return_final_only:
            final_step = jnp.asarray(self.recurrent_steps - 1, dtype=jnp.int32)
            logits = self._belief_to_token_logits(belief_final, tokens, final_step)
            diagnostics = {
                "diffusion_filled_ratio": jnp.zeros((self.recurrent_steps,), dtype=jnp.float32),
                "step_gate_mean": jnp.asarray(0.0, dtype=jnp.float32),
                "step_gate_std": jnp.asarray(0.0, dtype=jnp.float32),
                "denoise_energy": final_energy,
                "belief_kl_delta": final_kl_delta,
                "belief_entropy": final_entropy,
                "belief_confidence": final_confidence,
                "belief_update_norm": final_update_norm,
                "context_weight_reg": final_context_weight_reg,
                "context_weight_mean": final_context_weight_mean,
                "unroll_steps": jnp.asarray(self.recurrent_steps, dtype=jnp.float32),
                "draft": jnp.argmax(belief_final, axis=-1).astype(jnp.int32),
                "belief_logits": belief_final,
            }
            if return_raw_final_logits:
                diagnostics["raw_final_logits"] = raw_final_logits
            return logits, diagnostics
        (
            step_logits,
            filled_ratio,
            gate_mean,
            gate_std,
            energy,
            kl_delta,
            entropy,
            confidence,
            update_norm,
            context_weight_reg,
            context_weight_mean,
        ) = scan_outputs
        diagnostics = {
            "diffusion_filled_ratio": filled_ratio,
            "step_gate_mean": jnp.mean(gate_mean),
            "step_gate_std": jnp.mean(gate_std),
            "denoise_energy": jnp.mean(energy),
            "belief_kl_delta": jnp.mean(kl_delta),
            "belief_entropy": jnp.mean(entropy),
            "belief_confidence": jnp.mean(confidence),
            "belief_update_norm": jnp.mean(update_norm),
            "context_weight_reg": jnp.mean(context_weight_reg),
            "context_weight_mean": jnp.mean(context_weight_mean),
            "per_step_denoise_energy": energy,
            "per_step_belief_entropy": entropy,
            "unroll_steps": jnp.asarray(self.recurrent_steps, dtype=jnp.float32),
            "draft": jnp.argmax(belief_final, axis=-1).astype(jnp.int32),
            "belief_logits": belief_final,
        }
        if return_raw_final_logits:
            diagnostics["raw_final_logits"] = raw_final_logits
        return step_logits, diagnostics

    def forward_all_steps_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        initial_draft: Array | None = None,
        initial_belief: Array | None = None,
        compute_terminal_residual: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        del compute_terminal_residual
        return self.run_diffusion(
            tokens,
            initial_draft=initial_draft,
            initial_belief=initial_belief,
            train=train,
            dropout_key=dropout_key,
        )

    def forward_final_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        initial_draft: Array | None = None,
        initial_belief: Array | None = None,
        compute_terminal_residual: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        return self.forward_all_steps_with_diagnostics(
            tokens,
            train=train,
            dropout_key=dropout_key,
            initial_draft=initial_draft,
            initial_belief=initial_belief,
            compute_terminal_residual=compute_terminal_residual,
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
