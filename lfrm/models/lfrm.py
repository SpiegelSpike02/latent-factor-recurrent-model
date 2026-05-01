from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from lfrm.config import ModelConfig, RuntimeConfig
from .common import Array, compute_dtype, maybe_cast


State = tuple[Array, Array, Array, Array]


def _safe_log(x: Array, floor: float) -> Array:
    return jnp.log(jnp.clip(x, floor, 1.0))


def _contract(subscripts: str, *operands: Array) -> Array:
    return jnp.einsum(subscripts, *operands, preferred_element_type=jnp.float32)


class LatentFactorRecurrentModel(nnx.Module):
    """Task-agnostic latent factor recurrent model.

    LFRM uses a Perceiver IO-style latent bottleneck as its base: latent
    factors read from grid cells, the latent array is processed recurrently,
    and cells read back from those factors to refine a belief canvas. It
    deliberately avoids task-specific Sudoku row/column/box structure.
    Digits/symbols are represented as an exchangeable belief axis; no
    symbol-specific embeddings are used.
    """

    def __init__(self, config: ModelConfig, runtime: RuntimeConfig, *, rngs: nnx.Rngs) -> None:
        if config.model_type != "lfrm":
            raise ValueError("LatentFactorRecurrentModel requires model_type='lfrm'")
        if config.grid_height * config.grid_width != config.seq_len:
            raise ValueError("grid_height * grid_width must equal seq_len")
        if config.num_steps < 1:
            raise ValueError("LFRM num_steps must be at least 1")
        lfrm = config.lfrm_config
        belief_dim = lfrm.belief_dim or (config.vocab_size - 2)
        if belief_dim != config.vocab_size - 2:
            raise ValueError("LFRM expects belief_dim == vocab_size - 2 for grid classification tokens")
        if lfrm.num_slots < 1:
            raise ValueError("LFRM num_slots must be at least 1")
        if lfrm.num_branches < 1:
            raise ValueError("LFRM num_branches must be at least 1")
        if lfrm.num_heads < 1:
            raise ValueError("LFRM num_heads must be at least 1")
        if lfrm.latent_processor_layers < 0:
            raise ValueError("LFRM latent_processor_layers must be non-negative")
        if config.d_model % lfrm.num_heads != 0:
            raise ValueError("LFRM requires d_model to be divisible by num_heads")
        if lfrm.symbol_context_mode != "cell_symbol_tokens":
            raise ValueError("LFRM v1 requires symbol_context_mode='cell_symbol_tokens'")
        if lfrm.slot_readout_mode != "cell_symbol_attention":
            raise ValueError("LFRM v1 requires slot_readout_mode='cell_symbol_attention'")
        if lfrm.energy_symbol_pooling != "deepsets":
            raise ValueError("LFRM v1 requires energy_symbol_pooling='deepsets'")
        if lfrm.branch_diversity_schedule not in ("early", "none"):
            raise ValueError("LFRM branch_diversity_schedule must be 'early' or 'none'")

        self.config = config
        self.runtime = runtime
        self.lfrm = lfrm
        self.belief_dim = belief_dim
        self.head_dim = config.d_model // lfrm.num_heads
        dtype = compute_dtype(runtime.compute_dtype)
        self.dtype = dtype

        self.row_ids = jnp.repeat(jnp.arange(config.grid_height, dtype=jnp.int32), config.grid_width)
        self.col_ids = jnp.tile(jnp.arange(config.grid_width, dtype=jnp.int32), config.grid_height)
        self.branch_ids = jnp.arange(lfrm.num_branches, dtype=jnp.int32)
        self.slot_ids = jnp.arange(lfrm.num_slots, dtype=jnp.int32)

        d_model = config.d_model
        self.row_embed = nnx.Embed(config.grid_height, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.col_embed = nnx.Embed(config.grid_width, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.condition_type_embed = nnx.Embed(2, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.branch_embed = nnx.Embed(lfrm.num_branches, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.slot_embed = nnx.Embed(lfrm.num_slots, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

        local_dim = 2 * d_model + 2
        self.local_depthwise = nnx.Conv(
            d_model,
            d_model,
            kernel_size=(3, 3),
            padding="SAME",
            feature_group_count=d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            preferred_element_type=jnp.float32,
            rngs=rngs,
        )
        self.local_norm = nnx.RMSNorm(local_dim, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.local_gate = nnx.Linear(local_dim, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.local_candidate = nnx.Linear(local_dim, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

        micro_token_dim = d_model + 5
        self.micro_token_norm = nnx.RMSNorm(micro_token_dim, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.micro_token_hidden = nnx.Linear(micro_token_dim, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.symbol_context_norm = nnx.RMSNorm(2 * d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.symbol_context_hidden = nnx.Linear(2 * d_model, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.input_key = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.input_symbol_value = nnx.Linear(d_model, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.latent_input_query = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

        slot_update_dim = 2 * d_model + 1
        self.slot_update_norm = nnx.RMSNorm(slot_update_dim, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.slot_gate = nnx.Linear(slot_update_dim, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.slot_candidate = nnx.Linear(slot_update_dim, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

        self.latent_norm = nnx.RMSNorm(d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.latent_query = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.latent_key = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.latent_value = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.latent_output = nnx.Linear(d_model, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.latent_ff_norm = nnx.RMSNorm(d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.latent_ff_hidden = nnx.Linear(d_model, 2 * d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.latent_ff_output = nnx.Linear(2 * d_model, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

        self.output_query = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.latent_output_key = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.latent_output_value = nnx.Linear(d_model, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

        cell_update_dim = 3 * d_model + 2
        self.cell_update_norm = nnx.RMSNorm(cell_update_dim, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.cell_gate = nnx.Linear(cell_update_dim, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.cell_candidate = nnx.Linear(cell_update_dim, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

        symbol_dim = 3 * d_model + 4
        self.symbol_norm = nnx.RMSNorm(symbol_dim, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.symbol_hidden = nnx.Linear(symbol_dim, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.symbol_delta = nnx.Linear(d_model, 1, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

        energy_hidden = lfrm.energy_hidden_dim
        self.energy_cell_feature = nnx.Linear(2 * d_model + 3, energy_hidden, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.energy_cell_score = nnx.Linear(energy_hidden, 1, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.energy_slot_score = nnx.Linear(d_model, 1, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

    def _split_heads(self, x: Array) -> Array:
        return x.reshape(*x.shape[:-1], self.lfrm.num_heads, self.head_dim)

    def _merge_heads(self, x: Array) -> Array:
        return x.reshape(*x.shape[:-2], self.config.d_model)

    def _condition_mask(self, tokens: Array) -> Array:
        return tokens != 1

    def _belief_stats(self, q: Array) -> Array:
        entropy = -jnp.sum(q * _safe_log(q, self.lfrm.belief_floor), axis=-1, keepdims=True)
        entropy = entropy / max(math.log(self.belief_dim), 1e-6)
        confidence = jnp.max(q, axis=-1, keepdims=True)
        return jnp.concatenate([entropy, confidence], axis=-1)

    def _initial_beliefs_and_logits(self, tokens: Array, condition_mask: Array) -> tuple[Array, Array]:
        digit_ids = jnp.clip(tokens - 2, 0, self.belief_dim - 1)
        floor = jnp.minimum(jnp.asarray(self.lfrm.belief_floor, dtype=jnp.float32), 0.5 / self.belief_dim)
        given_q = jax.nn.one_hot(digit_ids, self.belief_dim, dtype=jnp.float32)
        given_q = given_q * (1.0 - self.belief_dim * floor) + floor
        blank_q = jnp.full_like(given_q, 1.0 / self.belief_dim)
        q = jnp.where(condition_mask[..., None], given_q, blank_q)
        logits = _safe_log(q, self.lfrm.belief_floor)
        logits = logits - jnp.mean(logits, axis=-1, keepdims=True)
        return q, logits

    def _clamp_logits(self, logits: Array, initial_logits: Array, condition_mask: Array) -> Array:
        logits = logits - jnp.mean(logits, axis=-1, keepdims=True)
        return jnp.where(condition_mask[:, None, :, None], initial_logits[:, None, :, :], logits)

    def _initial_state(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> tuple[State, Array, Array, Array]:
        condition_mask = self._condition_mask(tokens)
        q0, logits0 = self._initial_beliefs_and_logits(tokens, condition_mask)

        base_hidden = self.row_embed(self.row_ids)[None, :, :]
        base_hidden = base_hidden + self.col_embed(self.col_ids)[None, :, :]
        condition_type = self.condition_type_embed(condition_mask.astype(jnp.int32))
        if self.lfrm.use_condition_type_embedding:
            base_hidden = base_hidden + condition_type
        base_hidden = self.dropout(base_hidden, deterministic=not train, rngs=dropout_key)
        branch_hidden = base_hidden[:, None, :, :] + self.branch_embed(self.branch_ids)[None, :, None, :]
        branch_hidden = branch_hidden.astype(jnp.float32)

        q = jnp.broadcast_to(
            q0[:, None, :, :],
            (tokens.shape[0], self.lfrm.num_branches, self.config.seq_len, self.belief_dim),
        )
        logits = jnp.broadcast_to(
            logits0[:, None, :, :],
            (tokens.shape[0], self.lfrm.num_branches, self.config.seq_len, self.belief_dim),
        )
        slots = self.slot_embed(self.slot_ids)[None, None, :, :]
        slots = slots + self.branch_embed(self.branch_ids)[None, :, None, :]
        slots = jnp.broadcast_to(
            slots.astype(jnp.float32),
            (
                tokens.shape[0],
                self.lfrm.num_branches,
                self.lfrm.num_slots,
                self.config.d_model,
            ),
        )
        return (branch_hidden, logits, q, slots), logits0, q0, condition_mask

    def _local_mix(self, h: Array) -> Array:
        batch, branches, _, channels = h.shape
        grid = h.reshape(batch * branches, self.config.grid_height, self.config.grid_width, channels)
        local = self.local_depthwise(maybe_cast(grid, self.dtype)).astype(jnp.float32)
        return local.reshape(batch, branches, self.config.seq_len, channels)

    def _local_update(self, h: Array, q: Array) -> Array:
        stats = self._belief_stats(q)
        local_input = jnp.concatenate([h, self._local_mix(h), stats], axis=-1)
        local_input = self.local_norm(maybe_cast(local_input, self.dtype))
        gate = jax.nn.sigmoid(self.local_gate(local_input).astype(jnp.float32))
        candidate = jnp.tanh(self.local_candidate(local_input).astype(jnp.float32))
        return h.astype(jnp.float32) + 0.1 * gate * candidate

    def _given_channels(self, initial_q: Array, condition_mask: Array, branches: int) -> Array:
        given = jnp.where(condition_mask[..., None], initial_q, 0.0)
        return jnp.broadcast_to(
            given[:, None, :, :],
            (given.shape[0], branches, self.config.seq_len, self.belief_dim),
        )

    def _cell_symbol_context(self, h: Array, logits: Array, q: Array, given_channels: Array) -> tuple[Array, Array]:
        h_symbols = jnp.broadcast_to(h[:, :, :, None, :], (*q.shape, h.shape[-1]))
        stats = self._belief_stats(q)
        stats_symbols = jnp.broadcast_to(stats[:, :, :, None, :], (*q.shape, stats.shape[-1]))
        symbol_input = jnp.concatenate(
            [h_symbols, q[..., None], logits[..., None], given_channels[..., None], stats_symbols],
            axis=-1,
        )
        symbol_input = self.micro_token_norm(maybe_cast(symbol_input, self.dtype))
        base_tokens = jax.nn.silu(self.micro_token_hidden(symbol_input).astype(jnp.float32))
        symbol_context = jnp.mean(base_tokens, axis=2)
        context_tokens = jnp.broadcast_to(symbol_context[:, :, None, :, :], base_tokens.shape)
        token_input = jnp.concatenate([base_tokens, context_tokens], axis=-1)
        token_input = self.symbol_context_norm(maybe_cast(token_input, self.dtype))
        tokens = jax.nn.silu(self.symbol_context_hidden(token_input).astype(jnp.float32))
        return tokens, symbol_context

    def _slot_pattern_decorrelation(self, attention: Array) -> Array:
        patterns = jnp.mean(attention, axis=2)
        patterns = patterns.reshape(*patterns.shape[:3], self.config.seq_len * self.belief_dim)
        patterns = patterns / jnp.maximum(jnp.linalg.norm(patterns, axis=-1, keepdims=True), self.lfrm.belief_floor)
        cosine = _contract("brmt,brnt->brmn", patterns, patterns).astype(jnp.float32)
        offdiag = 1.0 - jnp.eye(self.lfrm.num_slots, dtype=jnp.float32)
        return jnp.sum(cosine * offdiag[None, None, :, :]) / jnp.maximum(
            jnp.asarray(cosine.shape[0] * cosine.shape[1] * self.lfrm.num_slots * (self.lfrm.num_slots - 1), dtype=jnp.float32),
            1.0,
        )

    def _cells_to_slots(self, micro_tokens: Array, slots: Array) -> tuple[Array, Array, Array, Array]:
        input_key = self._split_heads(self.input_key(maybe_cast(micro_tokens, self.dtype)))
        latent_query = self._split_heads(self.latent_input_query(maybe_cast(slots, self.dtype)))
        scores = _contract("brmhd,brnkhd->brhmnk", latent_query, input_key)
        scores = scores.astype(jnp.float32) / math.sqrt(self.head_dim)
        scores = scores / self.lfrm.assignment_temperature
        flat_scores = scores.reshape(*scores.shape[:4], self.config.seq_len * self.belief_dim)
        input_attention = jax.nn.softmax(flat_scores, axis=-1).reshape(scores.shape)
        symbol_values = self._split_heads(self.input_symbol_value(maybe_cast(micro_tokens, self.dtype)))
        summary = _contract(
            "brhmnk,brnkhd->brmhd",
            maybe_cast(input_attention, self.dtype),
            symbol_values,
        )
        summary = self._merge_heads(summary)
        assignment = jnp.mean(input_attention, axis=2)
        assignment = assignment.reshape(*assignment.shape[:3], self.config.seq_len * self.belief_dim)
        slot_focus = jnp.mean(jnp.max(input_attention.reshape(*input_attention.shape[:4], -1), axis=-1), axis=2)
        slot_diversity_loss = self._slot_pattern_decorrelation(input_attention)
        return summary, assignment, slot_focus, slot_diversity_loss

    def _update_slots(self, slots: Array, summary: Array, slot_usage: Array) -> Array:
        usage = slot_usage[:, :, :, None]
        slot_input = jnp.concatenate([slots.astype(jnp.float32), summary, usage], axis=-1)
        slot_input = self.slot_update_norm(maybe_cast(slot_input, self.dtype))
        gate = jax.nn.sigmoid(self.slot_gate(slot_input).astype(jnp.float32))
        candidate = jnp.tanh(self.slot_candidate(slot_input).astype(jnp.float32))
        return slots.astype(jnp.float32) + 0.1 * gate * candidate

    def _process_latents(self, slots: Array) -> Array:
        for _ in range(self.lfrm.latent_processor_layers):
            latent_input = self.latent_norm(maybe_cast(slots, self.dtype))
            query = self._split_heads(self.latent_query(latent_input))
            key = self._split_heads(self.latent_key(latent_input))
            value = self._split_heads(self.latent_value(latent_input))
            scores = _contract("brmhd,brnhd->brhmn", query, key)
            scores = scores.astype(jnp.float32) / math.sqrt(self.head_dim)
            attention = jax.nn.softmax(scores, axis=-1)
            message = _contract("brhmn,brnhd->brmhd", maybe_cast(attention, self.dtype), value)
            message = self._merge_heads(message)
            slots = slots.astype(jnp.float32) + 0.1 * self.latent_output(maybe_cast(message, self.dtype)).astype(jnp.float32)
            ff_input = self.latent_ff_norm(maybe_cast(slots, self.dtype))
            ff_hidden = jax.nn.silu(self.latent_ff_hidden(ff_input).astype(jnp.float32))
            slots = slots.astype(jnp.float32) + 0.1 * self.latent_ff_output(maybe_cast(ff_hidden, self.dtype)).astype(jnp.float32)
        return slots

    def _slots_to_cell_symbols(self, micro_tokens: Array, slots: Array) -> tuple[Array, Array]:
        cell_query = self._split_heads(self.output_query(maybe_cast(micro_tokens, self.dtype)))
        slot_key = self._split_heads(self.latent_output_key(maybe_cast(slots, self.dtype)))
        scores = _contract("brnkhd,brmhd->brhnkm", cell_query, slot_key)
        scores = scores.astype(jnp.float32) / math.sqrt(self.head_dim)
        scores = scores / self.lfrm.assignment_temperature
        routing = jax.nn.softmax(scores, axis=-1)
        slot_value = self._split_heads(self.latent_output_value(maybe_cast(slots, self.dtype)))
        symbol_message = _contract("brhnkm,brmhd->brnkhd", maybe_cast(routing, self.dtype), slot_value)
        symbol_message = self._merge_heads(symbol_message)
        routing = jnp.mean(routing, axis=2)
        return symbol_message, routing

    def _update_state(
        self,
        state: State,
        *,
        initial_logits: Array,
        initial_q: Array,
        initial_h: Array,
        condition_mask: Array,
    ) -> tuple[State, dict[str, Array]]:
        h, logits, q, slots = state
        h_local = self._local_update(h, q)
        given_channels = self._given_channels(initial_q, condition_mask, h.shape[1])
        micro_tokens, symbol_context = self._cell_symbol_context(h_local, logits, q, given_channels)
        slot_summary, assignment, input_slot_usage, slot_diversity_loss = self._cells_to_slots(micro_tokens, slots)
        slots_next = self._update_slots(slots, slot_summary, input_slot_usage)
        slots_next = self._process_latents(slots_next)
        symbol_message, _ = self._slots_to_cell_symbols(micro_tokens, slots_next)
        cell_message = jnp.sum(symbol_message * q[..., None], axis=-2)
        cell_stats = self._belief_stats(q)
        cell_input = jnp.concatenate([h, h_local, cell_message, cell_stats], axis=-1)
        cell_input = self.cell_update_norm(maybe_cast(cell_input, self.dtype))
        cell_gate = jax.nn.sigmoid(self.cell_gate(cell_input).astype(jnp.float32))
        cell_candidate = jnp.tanh(self.cell_candidate(cell_input).astype(jnp.float32))
        h_next = h.astype(jnp.float32) + 0.1 * cell_gate * cell_candidate
        if self.lfrm.freeze_conditioned_state:
            h_next = jnp.where(condition_mask[:, None, :, None], initial_h, h_next)

        h_symbols = jnp.broadcast_to(h_next[:, :, :, None, :], symbol_message.shape)
        stats_symbols = jnp.broadcast_to(cell_stats[:, :, :, None, :], (*q.shape, cell_stats.shape[-1]))
        symbol_input = jnp.concatenate(
            [h_symbols, symbol_message, micro_tokens, q[..., None], logits[..., None], stats_symbols],
            axis=-1,
        )
        symbol_input = self.symbol_norm(maybe_cast(symbol_input, self.dtype))
        symbol_hidden = jax.nn.silu(self.symbol_hidden(symbol_input).astype(jnp.float32))
        delta = self.symbol_delta(maybe_cast(symbol_hidden, self.dtype)).astype(jnp.float32).squeeze(-1)
        delta = delta - jnp.mean(delta, axis=-1, keepdims=True)
        logits_next = logits + self.lfrm.belief_step_size * delta
        logits_next = self._clamp_logits(logits_next, initial_logits, condition_mask)
        q_next = jax.nn.softmax(logits_next / self.lfrm.belief_temperature, axis=-1)
        q_next = jnp.where(condition_mask[:, None, :, None], initial_q[:, None, :, :], q_next)
        return (h_next, logits_next, q_next, slots_next), {
            "assignment": assignment,
            "slot_usage": input_slot_usage,
            "slot_diversity_loss": slot_diversity_loss,
            "symbol_context": symbol_context,
        }

    def _energy(self, q: Array, h: Array, slots: Array, symbol_context: Array) -> Array:
        h_symbols = jnp.broadcast_to(h[:, :, :, None, :], (*q.shape, h.shape[-1]))
        context_symbols = jnp.broadcast_to(symbol_context[:, :, None, :, :], h_symbols.shape)
        stats = self._belief_stats(q)
        stats_symbols = jnp.broadcast_to(stats[:, :, :, None, :], (*q.shape, stats.shape[-1]))
        channel_features = jnp.concatenate(
            [h_symbols.astype(jnp.float32), context_symbols.astype(jnp.float32), q[..., None], stats_symbols],
            axis=-1,
        )
        channel_features = jax.nn.silu(
            self.energy_cell_feature(maybe_cast(channel_features, self.dtype)).astype(jnp.float32)
        )
        cell_features = jnp.mean(channel_features, axis=-2)
        cell_energy = jnp.mean(
            self.energy_cell_score(maybe_cast(cell_features, self.dtype)).astype(jnp.float32).squeeze(-1),
            axis=-1,
        )
        slot_energy = jnp.mean(
            self.energy_slot_score(maybe_cast(slots, self.dtype)).astype(jnp.float32).squeeze(-1),
            axis=2,
        )
        return cell_energy + slot_energy

    def _branch_logits_to_vocab(self, logits: Array) -> Array:
        low_logits = jnp.full((*logits.shape[:-1], 2), -30.0, dtype=jnp.float32)
        return jnp.concatenate([low_logits, logits.astype(jnp.float32)], axis=-1)

    def _blank_mean(self, values: Array, condition_mask: Array) -> Array:
        mask = (~condition_mask).astype(jnp.float32)[:, None, :, None]
        trailing_size = math.prod(values.shape[3:]) if len(values.shape) > 3 else 1
        normalizer = jnp.maximum(jnp.sum(mask) * values.shape[1] * trailing_size, 1.0)
        return jnp.sum(values * mask) / normalizer

    def _run_unroll(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None,
        compute_energy_selection: bool = True,
        compute_terminal_residual: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        state0, initial_logits, initial_q, condition_mask = self._initial_state(tokens, train=train, dropout_key=dropout_key)
        initial_h = state0[0]
        empty_assignment = jnp.zeros(
            (
                tokens.shape[0],
                self.lfrm.num_branches,
                self.lfrm.num_slots,
                self.config.seq_len * self.belief_dim,
            ),
            dtype=jnp.float32,
        )

        def update_step(state: State) -> tuple[State, dict[str, Array]]:
            return self._update_state(
                state,
                initial_logits=initial_logits,
                initial_q=initial_q,
                initial_h=initial_h,
                condition_mask=condition_mask,
            )

        remat_update_step = jax.checkpoint(update_step)

        def blank_branch_diversity(q: Array) -> Array:
            variance = jnp.var(q, axis=1, keepdims=True)
            return self._blank_mean(variance, condition_mask)

        def scan_step(carry, step_index):
            (
                state,
                prev_assignment,
                consistency_total,
                usage_entropy_total,
                slot_diversity_total,
                branch_diversity_total,
                branch_diversity_count,
                hidden_delta,
            ) = carry
            h_prev = state[0]
            next_state, aux = remat_update_step(state)
            assignment = aux["assignment"]
            prev = jnp.clip(prev_assignment, self.lfrm.belief_floor, 1.0)
            curr = jnp.clip(assignment, self.lfrm.belief_floor, 1.0)
            step_kl = jnp.mean(
                jnp.sum(
                    jax.lax.stop_gradient(prev)
                    * (_safe_log(prev, self.lfrm.belief_floor) - _safe_log(curr, self.lfrm.belief_floor)),
                    axis=-1,
                )
            )
            step_kl = jnp.where(step_index > 0, step_kl, 0.0)

            usage = jnp.clip(aux["slot_usage"], self.lfrm.belief_floor, 1.0)
            usage = usage / jnp.maximum(jnp.sum(usage, axis=-1, keepdims=True), self.lfrm.belief_floor)
            usage_entropy = -jnp.sum(usage * _safe_log(usage, self.lfrm.belief_floor), axis=-1)
            usage_entropy = jnp.mean(usage_entropy / max(math.log(self.lfrm.num_slots), 1e-6))

            h_delta = jnp.linalg.norm((next_state[0] - h_prev).astype(jnp.float32), axis=-1, keepdims=True)
            hidden_delta = self._blank_mean(h_delta, condition_mask)
            diversity_start = jnp.asarray(self.lfrm.diversity_apply_steps[0], dtype=step_index.dtype)
            diversity_end = jnp.asarray(self.lfrm.diversity_apply_steps[1], dtype=step_index.dtype)
            use_branch_diversity = (
                (self.lfrm.branch_diversity_schedule == "early")
                & (step_index >= diversity_start)
                & (step_index < diversity_end)
            )
            branch_diversity_step = blank_branch_diversity(next_state[2])
            branch_diversity_mask = use_branch_diversity.astype(jnp.float32)
            return (
                next_state,
                assignment,
                consistency_total + step_kl,
                usage_entropy_total + usage_entropy,
                slot_diversity_total + aux["slot_diversity_loss"],
                branch_diversity_total + branch_diversity_mask * branch_diversity_step,
                branch_diversity_count + branch_diversity_mask,
                hidden_delta,
            ), None

        initial_carry = (
            state0,
            empty_assignment,
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        (
            final_state,
            _,
            consistency_total,
            usage_entropy_total,
            slot_diversity_total,
            branch_diversity_total,
            branch_diversity_count,
            hidden_delta,
        ), _ = jax.lax.scan(
            scan_step,
            initial_carry,
            jnp.arange(self.config.num_steps),
        )
        h_final, logits_final, q_final, slots_final = final_state
        selected_index = jnp.zeros((tokens.shape[0],), dtype=jnp.int32)
        branch_energy = None
        symbol_context_final = None
        if compute_energy_selection:
            final_given_channels = self._given_channels(initial_q, condition_mask, h_final.shape[1])
            _, symbol_context_final = self._cell_symbol_context(h_final, logits_final, q_final, final_given_channels)
            branch_energy = self._energy(q_final, h_final, slots_final, symbol_context_final)
            selected_index = jnp.argmin(branch_energy, axis=1)
        selected_index_expanded = selected_index[:, None, None, None]
        selected_digit_logits = jnp.take_along_axis(
            logits_final,
            jnp.broadcast_to(selected_index_expanded, (tokens.shape[0], 1, self.config.seq_len, self.belief_dim)),
            axis=1,
        ).squeeze(1)
        selected_logits = self._branch_logits_to_vocab(selected_digit_logits)
        selected_q = jnp.take_along_axis(
            q_final,
            jnp.broadcast_to(selected_index_expanded, (tokens.shape[0], 1, self.config.seq_len, self.belief_dim)),
            axis=1,
        ).squeeze(1)

        entropy = -jnp.sum(selected_q * _safe_log(selected_q, self.lfrm.belief_floor), axis=-1, keepdims=True)
        blank_mask = (~condition_mask).astype(jnp.float32)[..., None]
        blank_normalizer = jnp.maximum(jnp.sum(blank_mask), 1.0)
        belief_entropy = jnp.sum(entropy * blank_mask) / blank_normalizer
        confidence = jnp.max(selected_q, axis=-1, keepdims=True)
        belief_confidence = jnp.sum(confidence * blank_mask) / blank_normalizer
        final_branch_diversity = blank_branch_diversity(q_final)
        branch_diversity = jnp.where(
            branch_diversity_count > 0.0,
            branch_diversity_total / branch_diversity_count,
            final_branch_diversity,
        )
        steps_for_consistency = jnp.maximum(jnp.asarray(self.config.num_steps - 1, dtype=jnp.float32), 1.0)
        slot_consistency_loss = consistency_total / steps_for_consistency
        slot_usage_entropy = usage_entropy_total / jnp.asarray(self.config.num_steps, dtype=jnp.float32)
        slot_diversity_loss = slot_diversity_total / jnp.asarray(self.config.num_steps, dtype=jnp.float32)

        diagnostics = {
            "hidden_delta_mean": jnp.asarray([hidden_delta], dtype=jnp.float32),
            "unroll_steps": jnp.asarray(self.config.num_steps, dtype=jnp.float32),
            "belief_entropy": belief_entropy,
            "belief_confidence": belief_confidence,
            "branch_digit_logits": logits_final,
            "branch_q": q_final,
            "branch_h": h_final,
            "branch_slots": slots_final,
            "slot_consistency_loss": slot_consistency_loss,
            "slot_usage_entropy": slot_usage_entropy,
            "slot_usage_loss": 1.0 - slot_usage_entropy,
            "slot_diversity_loss": slot_diversity_loss,
            "branch_diversity": branch_diversity,
            "final_branch_diversity": final_branch_diversity,
        }
        if compute_terminal_residual:
            next_state, _ = update_step(final_state)
            q_delta = next_state[2] - q_final
            q_residual_mse = self._blank_mean(jnp.square(q_delta), condition_mask)
            diagnostics["terminal_belief_mse"] = q_residual_mse
            diagnostics["terminal_belief_delta"] = jnp.sqrt(jnp.maximum(q_residual_mse, 0.0))
        if branch_energy is not None and symbol_context_final is not None:
            diagnostics["branch_logits"] = self._branch_logits_to_vocab(logits_final)
            diagnostics["branch_symbol_context"] = symbol_context_final
            diagnostics["branch_energy"] = branch_energy
            diagnostics["selected_branch_energy"] = jnp.mean(
                jnp.take_along_axis(branch_energy, selected_index[:, None], axis=1).squeeze(1)
            )
        return selected_logits[None, :, :, :], diagnostics

    def forward_all_steps_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        compute_energy_selection: bool = True,
        compute_terminal_residual: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        return self._run_unroll(
            tokens,
            train=train,
            dropout_key=dropout_key,
            compute_energy_selection=compute_energy_selection,
            compute_terminal_residual=compute_terminal_residual,
        )

    def forward_final_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
        compute_energy_selection: bool = True,
        compute_terminal_residual: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        return self.forward_all_steps_with_diagnostics(
            tokens,
            train=train,
            dropout_key=dropout_key,
            compute_energy_selection=compute_energy_selection,
            compute_terminal_residual=compute_terminal_residual,
        )

    def forward_all_steps(self, tokens: Array, *, train: bool, dropout_key: Array | None = None) -> Array:
        logits, _ = self.forward_all_steps_with_diagnostics(tokens, train=train, dropout_key=dropout_key)
        return logits

    def forward_final(self, tokens: Array, *, train: bool, dropout_key: Array | None = None) -> Array:
        logits, _ = self.forward_final_with_diagnostics(tokens, train=train, dropout_key=dropout_key)
        return logits

    def __call__(self, tokens: Array, *, train: bool, dropout_key: Array | None = None) -> Array:
        return self.forward_final(tokens, train=train, dropout_key=dropout_key)[-1]

    def energy_training_metrics(
        self,
        inputs: Array,
        targets: Array,
        given_mask: Array,
        rng_key: Array,
        diagnostics: dict[str, Array],
        *,
        margin: float,
        corruptions: int,
    ) -> dict[str, Array]:
        del inputs
        target_digits = jnp.clip(targets - 2, 0, self.belief_dim - 1)
        q_pos = jax.nn.one_hot(target_digits, self.belief_dim, dtype=jnp.float32)
        q_pos = jnp.broadcast_to(q_pos[:, None, :, :], diagnostics["branch_q"].shape)
        h = diagnostics["branch_h"]
        slots = diagnostics["branch_slots"]
        symbol_context = diagnostics["branch_symbol_context"]
        energy_pos = self._energy(q_pos, h, slots, symbol_context)

        corruption_count = max(int(corruptions), 1)
        keys = jax.random.split(rng_key, corruption_count)
        blank_mask = ~given_mask

        def corrupt_one(key):
            cell_key, offset_key = jax.random.split(key)
            log_weights = jnp.where(blank_mask, 0.0, -1e9)
            cell_indices = jax.random.categorical(cell_key, log_weights, axis=-1)
            offsets = jax.random.randint(offset_key, target_digits.shape, minval=1, maxval=self.belief_dim)
            batch_indices = jnp.arange(targets.shape[0])
            current = target_digits[batch_indices, cell_indices]
            corrupt_digits = (current + offsets[batch_indices, cell_indices]) % self.belief_dim
            q_neg = q_pos[:, 0, :, :]
            q_neg = q_neg.at[batch_indices, cell_indices, :].set(
                jax.nn.one_hot(corrupt_digits, self.belief_dim, dtype=jnp.float32)
            )
            return q_neg

        q_neg = jax.vmap(corrupt_one)(keys)
        q_neg = jnp.broadcast_to(
            q_neg[:, :, None, :, :],
            (corruption_count, targets.shape[0], self.lfrm.num_branches, self.config.seq_len, self.belief_dim),
        )
        flat_q_neg = q_neg.reshape(corruption_count * targets.shape[0], self.lfrm.num_branches, self.config.seq_len, self.belief_dim)
        flat_h = jnp.broadcast_to(
            h[None, :, :, :, :],
            (corruption_count, *h.shape),
        ).reshape(corruption_count * targets.shape[0], *h.shape[1:])
        flat_slots = jnp.broadcast_to(
            slots[None, :, :, :, :],
            (corruption_count, *slots.shape),
        ).reshape(corruption_count * targets.shape[0], *slots.shape[1:])
        flat_symbol_context = jnp.broadcast_to(
            symbol_context[None, :, :, :, :],
            (corruption_count, *symbol_context.shape),
        ).reshape(corruption_count * targets.shape[0], *symbol_context.shape[1:])
        energy_corrupt = self._energy(flat_q_neg, flat_h, flat_slots, flat_symbol_context).reshape(
            corruption_count,
            targets.shape[0],
            self.lfrm.num_branches,
        )

        corrupt_margin = jnp.maximum(0.0, margin + energy_pos[None, :, :] - energy_corrupt)
        margin_loss = jnp.mean(corrupt_margin)
        return {
            "energy_margin_loss": margin_loss,
            "energy_pos": jnp.mean(energy_pos),
            "energy_neg": jnp.mean(energy_corrupt),
        }
