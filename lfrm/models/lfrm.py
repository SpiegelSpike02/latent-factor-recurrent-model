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
        if lfrm.latent_processor_layers < 0:
            raise ValueError("LFRM latent_processor_layers must be non-negative")

        self.config = config
        self.runtime = runtime
        self.lfrm = lfrm
        self.belief_dim = belief_dim
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
        self.local_norm = nnx.RMSNorm(local_dim, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.local_gate = nnx.Linear(local_dim, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.local_candidate = nnx.Linear(local_dim, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

        self.input_key = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.input_symbol_value = nnx.Linear(d_model + 2, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
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

        symbol_dim = 2 * d_model + 2
        self.symbol_norm = nnx.RMSNorm(symbol_dim, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.symbol_hidden = nnx.Linear(symbol_dim, d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.symbol_delta = nnx.Linear(d_model, 1, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

        energy_hidden = lfrm.energy_hidden_dim
        self.energy_cell_feature = nnx.Linear(d_model + 2, energy_hidden, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.energy_cell_score = nnx.Linear(energy_hidden, 1, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.energy_slot_score = nnx.Linear(d_model, 1, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

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
        slots = self.slot_embed(self.slot_ids)[None, None, :, None, :]
        slots = slots + self.branch_embed(self.branch_ids)[None, :, None, None, :]
        slots = jnp.broadcast_to(
            slots.astype(jnp.float32),
            (
                tokens.shape[0],
                self.lfrm.num_branches,
                self.lfrm.num_slots,
                self.belief_dim,
                self.config.d_model,
            ),
        )
        return (branch_hidden, logits, q, slots), logits0, q0, condition_mask

    def _local_mean(self, h: Array) -> Array:
        batch, branches, _, channels = h.shape
        grid = h.reshape(batch * branches, self.config.grid_height, self.config.grid_width, channels)
        kernel = jnp.full((3, 3, 1, channels), 1.0 / 9.0, dtype=grid.dtype)
        local = jax.lax.conv_general_dilated(
            grid,
            kernel,
            window_strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=channels,
        )
        return local.reshape(batch, branches, self.config.seq_len, channels)

    def _local_update(self, h: Array, q: Array) -> Array:
        stats = self._belief_stats(q)
        local_input = jnp.concatenate([h, self._local_mean(h), stats], axis=-1)
        local_input = self.local_norm(maybe_cast(local_input, self.dtype))
        gate = jax.nn.sigmoid(self.local_gate(local_input).astype(jnp.float32))
        candidate = jnp.tanh(self.local_candidate(local_input).astype(jnp.float32))
        return h.astype(jnp.float32) + 0.1 * gate * candidate

    def _cell_symbol_values(self, h: Array, logits: Array, q: Array) -> Array:
        h_symbols = jnp.broadcast_to(h[:, :, :, None, :], (*q.shape, h.shape[-1]))
        symbol_input = jnp.concatenate([h_symbols, q[..., None], logits[..., None]], axis=-1)
        return self.input_symbol_value(maybe_cast(symbol_input, self.dtype)).astype(jnp.float32)

    def _cells_to_slots(self, h: Array, logits: Array, q: Array, slots: Array) -> tuple[Array, Array]:
        input_key = self.input_key(maybe_cast(h, self.dtype)).astype(jnp.float32)
        latent_query = self.latent_input_query(maybe_cast(slots, self.dtype)).astype(jnp.float32)
        scores = jnp.einsum("brmkd,brnd->brmnk", latent_query, input_key)
        scores = scores / math.sqrt(self.config.d_model)
        scores = scores / self.lfrm.assignment_temperature
        input_attention = jax.nn.softmax(scores, axis=3)
        symbol_values = self._cell_symbol_values(h, logits, q)
        weighted_attention = input_attention * q[:, :, None, :, :]
        normalizer = jnp.maximum(jnp.sum(weighted_attention, axis=3)[..., None], 1e-3)
        summary = jnp.einsum("brmnk,brnkd->brmkd", weighted_attention, symbol_values) / normalizer
        slot_usage = jnp.mean(jnp.sum(weighted_attention, axis=3), axis=-1)
        return summary, slot_usage

    def _update_slots(self, slots: Array, summary: Array, slot_usage: Array) -> Array:
        usage = slot_usage[:, :, :, None, None]
        usage = jnp.broadcast_to(usage, (*slots.shape[:-1], 1))
        slot_input = jnp.concatenate([slots.astype(jnp.float32), summary, usage], axis=-1)
        slot_input = self.slot_update_norm(maybe_cast(slot_input, self.dtype))
        gate = jax.nn.sigmoid(self.slot_gate(slot_input).astype(jnp.float32))
        candidate = jnp.tanh(self.slot_candidate(slot_input).astype(jnp.float32))
        return slots.astype(jnp.float32) + 0.1 * gate * candidate

    def _process_latents(self, slots: Array) -> Array:
        for _ in range(self.lfrm.latent_processor_layers):
            latent_input = self.latent_norm(maybe_cast(slots, self.dtype))
            query = self.latent_query(latent_input).astype(jnp.float32)
            key = self.latent_key(latent_input).astype(jnp.float32)
            value = self.latent_value(latent_input).astype(jnp.float32)
            query_by_symbol = jnp.swapaxes(query, 2, 3)
            key_by_symbol = jnp.swapaxes(key, 2, 3)
            value_by_symbol = jnp.swapaxes(value, 2, 3)
            scores = jnp.einsum("brkmd,brknd->brkmn", query_by_symbol, key_by_symbol)
            scores = scores / math.sqrt(self.config.d_model)
            attention = jax.nn.softmax(scores, axis=-1)
            message = jnp.einsum("brkmn,brknd->brkmd", attention, value_by_symbol)
            message = jnp.swapaxes(message, 2, 3)
            slots = slots.astype(jnp.float32) + 0.1 * self.latent_output(maybe_cast(message, self.dtype)).astype(jnp.float32)
            ff_input = self.latent_ff_norm(maybe_cast(slots, self.dtype))
            ff_hidden = jax.nn.silu(self.latent_ff_hidden(ff_input).astype(jnp.float32))
            slots = slots.astype(jnp.float32) + 0.1 * self.latent_ff_output(maybe_cast(ff_hidden, self.dtype)).astype(jnp.float32)
        return slots

    def _slots_to_cells(self, h: Array, slots: Array) -> tuple[Array, Array]:
        cell_query = self.output_query(maybe_cast(h, self.dtype)).astype(jnp.float32)
        slot_key = self.latent_output_key(maybe_cast(slots, self.dtype)).astype(jnp.float32)
        scores = jnp.einsum("brnd,brmkd->brnmk", cell_query, slot_key)
        scores = scores / math.sqrt(self.config.d_model)
        scores = scores / self.lfrm.assignment_temperature
        routing = jax.nn.softmax(scores, axis=3)
        slot_value = self.latent_output_value(maybe_cast(slots, self.dtype)).astype(jnp.float32)
        symbol_message = jnp.einsum("brnmk,brmkd->brnkd", routing, slot_value)
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
        slot_summary, input_slot_usage = self._cells_to_slots(h_local, logits, q, slots)
        slots_next = self._update_slots(slots, slot_summary, input_slot_usage)
        slots_next = self._process_latents(slots_next)
        symbol_message, routing = self._slots_to_cells(h_local, slots_next)
        output_slot_usage = jnp.mean(routing, axis=(2, 4))
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
        symbol_input = jnp.concatenate([h_symbols, symbol_message, q[..., None], logits[..., None]], axis=-1)
        symbol_input = self.symbol_norm(maybe_cast(symbol_input, self.dtype))
        symbol_hidden = jax.nn.silu(self.symbol_hidden(symbol_input).astype(jnp.float32))
        delta = self.symbol_delta(maybe_cast(symbol_hidden, self.dtype)).astype(jnp.float32).squeeze(-1)
        delta = delta - jnp.mean(delta, axis=-1, keepdims=True)
        logits_next = logits + self.lfrm.belief_step_size * delta
        logits_next = self._clamp_logits(logits_next, initial_logits, condition_mask)
        q_next = jax.nn.softmax(logits_next / self.lfrm.belief_temperature, axis=-1)
        q_next = jnp.where(condition_mask[:, None, :, None], initial_q[:, None, :, :], q_next)
        return (h_next, logits_next, q_next, slots_next), {
            "assignment": jnp.mean(routing, axis=-1),
            "slot_usage": output_slot_usage,
        }

    def _energy(self, q: Array, h: Array, slots: Array) -> Array:
        cell_features = jnp.concatenate([h.astype(jnp.float32), self._belief_stats(q)], axis=-1)
        cell_features = jax.nn.silu(self.energy_cell_feature(maybe_cast(cell_features, self.dtype)).astype(jnp.float32))
        cell_energy = jnp.mean(
            self.energy_cell_score(maybe_cast(cell_features, self.dtype)).astype(jnp.float32).squeeze(-1),
            axis=-1,
        )
        slot_energy = jnp.mean(
            self.energy_slot_score(maybe_cast(slots, self.dtype)).astype(jnp.float32).squeeze(-1),
            axis=(2, 3),
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
    ) -> tuple[Array, dict[str, Array]]:
        state0, initial_logits, initial_q, condition_mask = self._initial_state(tokens, train=train, dropout_key=dropout_key)
        initial_h = state0[0]
        empty_assignment = jnp.zeros(
            (
                tokens.shape[0],
                self.lfrm.num_branches,
                self.config.seq_len,
                self.lfrm.num_slots,
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

        def scan_step(carry, step_index):
            state, prev_assignment, consistency_total, usage_entropy_total, hidden_delta = carry
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
            return (
                next_state,
                assignment,
                consistency_total + step_kl,
                usage_entropy_total + usage_entropy,
                hidden_delta,
            ), None

        initial_carry = (
            state0,
            empty_assignment,
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        (final_state, _, consistency_total, usage_entropy_total, hidden_delta), _ = jax.lax.scan(
            scan_step,
            initial_carry,
            jnp.arange(self.config.num_steps),
        )
        h_final, logits_final, q_final, slots_final = final_state
        branch_energy = self._energy(q_final, h_final, slots_final)
        selected_index = jnp.argmin(branch_energy, axis=1)
        selected_index_expanded = selected_index[:, None, None, None]
        branch_vocab_logits = self._branch_logits_to_vocab(logits_final)
        selected_logits = jnp.take_along_axis(
            branch_vocab_logits,
            jnp.broadcast_to(selected_index_expanded, (tokens.shape[0], 1, self.config.seq_len, self.config.vocab_size)),
            axis=1,
        ).squeeze(1)
        selected_q = jnp.take_along_axis(
            q_final,
            jnp.broadcast_to(selected_index_expanded, (tokens.shape[0], 1, self.config.seq_len, self.belief_dim)),
            axis=1,
        ).squeeze(1)

        next_state, _ = update_step(final_state)
        q_delta = next_state[2] - q_final
        q_residual_mse = self._blank_mean(jnp.square(q_delta), condition_mask)
        q_residual_rms = jnp.sqrt(jnp.maximum(q_residual_mse, 0.0))
        entropy = -jnp.sum(selected_q * _safe_log(selected_q, self.lfrm.belief_floor), axis=-1, keepdims=True)
        blank_mask = (~condition_mask).astype(jnp.float32)[..., None]
        blank_normalizer = jnp.maximum(jnp.sum(blank_mask), 1.0)
        belief_entropy = jnp.sum(entropy * blank_mask) / blank_normalizer
        confidence = jnp.max(selected_q, axis=-1, keepdims=True)
        belief_confidence = jnp.sum(confidence * blank_mask) / blank_normalizer
        branch_diversity = jnp.mean(jnp.var(q_final, axis=1))
        steps_for_consistency = jnp.maximum(jnp.asarray(self.config.num_steps - 1, dtype=jnp.float32), 1.0)
        slot_consistency_loss = consistency_total / steps_for_consistency
        slot_usage_entropy = usage_entropy_total / jnp.asarray(self.config.num_steps, dtype=jnp.float32)
        selected_branch_energy = jnp.mean(
            jnp.take_along_axis(branch_energy, selected_index[:, None], axis=1).squeeze(1)
        )

        diagnostics = {
            "hidden_delta_mean": jnp.asarray([hidden_delta], dtype=jnp.float32),
            "unroll_steps": jnp.asarray(self.config.num_steps, dtype=jnp.float32),
            "terminal_belief_delta": q_residual_rms,
            "terminal_belief_mse": q_residual_mse,
            "belief_entropy": belief_entropy,
            "belief_confidence": belief_confidence,
            "branch_logits": branch_vocab_logits,
            "branch_digit_logits": logits_final,
            "branch_q": q_final,
            "branch_h": h_final,
            "branch_slots": slots_final,
            "branch_energy": branch_energy,
            "selected_branch_energy": selected_branch_energy,
            "slot_consistency_loss": slot_consistency_loss,
            "slot_usage_entropy": slot_usage_entropy,
            "slot_usage_loss": 1.0 - slot_usage_entropy,
            "branch_diversity": branch_diversity,
        }
        return selected_logits[None, :, :, :], diagnostics

    def forward_all_steps_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        return self._run_unroll(tokens, train=train, dropout_key=dropout_key)

    def forward_final_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        return self.forward_all_steps_with_diagnostics(tokens, train=train, dropout_key=dropout_key)

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
        energy_pos = self._energy(q_pos, h, slots)
        energy_model = diagnostics["branch_energy"]

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
            slots[None, :, :, :, :, :],
            (corruption_count, *slots.shape),
        ).reshape(corruption_count * targets.shape[0], *slots.shape[1:])
        energy_corrupt = self._energy(flat_q_neg, flat_h, flat_slots).reshape(corruption_count, targets.shape[0], self.lfrm.num_branches)

        model_margin = jnp.maximum(0.0, margin + energy_pos - energy_model)
        corrupt_margin = jnp.maximum(0.0, margin + energy_pos[None, :, :] - energy_corrupt)
        margin_loss = 0.5 * (jnp.mean(model_margin) + jnp.mean(corrupt_margin))
        return {
            "energy_margin_loss": margin_loss,
            "energy_pos": jnp.mean(energy_pos),
            "energy_neg": 0.5 * (jnp.mean(energy_model) + jnp.mean(energy_corrupt)),
        }
