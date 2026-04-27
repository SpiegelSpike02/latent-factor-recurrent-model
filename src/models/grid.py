from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from config import ModelConfig, RuntimeConfig
from tasks.sudoku import sudoku_box_ids, sudoku_num_box_units, sudoku_relation_matrices, sudoku_unit_matrices
from .common import Array, compute_dtype
from .embeddings import GridEmbeddings
from .recurrent_transformer import RecurrentTransformerBlock
from .universal_transformer import UniversalTransformerBlock


class GridReasoningModel(nnx.Module):
    """Shared grid wrapper for UT and recurrent Transformer experiments."""

    def __init__(self, config: ModelConfig, runtime: RuntimeConfig, *, rngs: nnx.Rngs) -> None:
        if config.grid_height * config.grid_width != config.seq_len:
            raise ValueError("grid_height * grid_width must equal seq_len")
        if config.model_type not in ("universal_transformer", "recurrent_transformer"):
            raise ValueError(f"Unsupported model_type: {config.model_type}")
        if config.transition_type not in ("residual", "damped"):
            raise ValueError(f"Unsupported transition.type: {config.transition_type}")
        if config.model_type == "recurrent_transformer" and config.uses_damped_transition:
            raise ValueError("transition.type = 'damped' is only supported for model_type='universal_transformer'")
        if config.inner_steps < 1:
            raise ValueError("compute.inner_steps must be at least 1")
        if config.layers_per_step < 1:
            raise ValueError("compute.layers_per_step must be at least 1")
        if config.grad_inner_steps < 1 or config.grad_inner_steps > config.inner_steps:
            raise ValueError("compute.grad_inner_steps must be in [1, compute.inner_steps]")

        self.config = config
        self.runtime = runtime
        dtype = compute_dtype(runtime.compute_dtype)
        self.step_embed = nnx.Embed(num_embeddings=config.num_steps, features=config.d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.final_norm = nnx.RMSNorm(config.d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.output_head = nnx.Linear(config.d_model, config.vocab_size, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

        row_ids = jnp.repeat(jnp.arange(config.grid_height, dtype=jnp.int32), config.grid_width)
        col_ids = jnp.tile(jnp.arange(config.grid_width, dtype=jnp.int32), config.grid_height)
        box_ids = sudoku_box_ids(row_ids, col_ids, grid_height=config.grid_height, grid_width=config.grid_width)
        num_box_units = sudoku_num_box_units(
            grid_height=config.grid_height,
            grid_width=config.grid_width,
            seq_len=config.seq_len,
        )
        self.row_ids = row_ids
        self.col_ids = col_ids
        self.box_ids = box_ids
        self.num_box_units = num_box_units
        self.step_ids = jnp.arange(config.num_steps, dtype=jnp.int32)
        self.inference_step_dropout_keys = jax.random.split(jax.random.key(0), 2 * config.num_steps).reshape(config.num_steps, 2)
        self.embeddings = GridEmbeddings(
            config,
            runtime,
            row_ids=row_ids,
            col_ids=col_ids,
            box_ids=box_ids,
            num_box_units=num_box_units,
            rngs=rngs,
        )

        self.row_relation, self.col_relation, self.box_relation, self.global_relation = self._build_communication_relations(
            row_ids,
            col_ids,
            box_ids,
        )
        self.row_unit_matrix, self.col_unit_matrix, self.box_unit_matrix = sudoku_unit_matrices(
            row_ids,
            col_ids,
            box_ids,
            grid_height=config.grid_height,
            grid_width=config.grid_width,
            num_box_units=num_box_units,
        )
        if config.model_type == "universal_transformer":
            self.blocks = nnx.List(
                UniversalTransformerBlock(
                    config,
                    runtime,
                    row_relation=self.row_relation,
                    col_relation=self.col_relation,
                    box_relation=self.box_relation,
                    global_relation=self.global_relation,
                    rngs=rngs,
                )
                for _ in range(config.layers_per_step)
            )
        else:
            self.blocks = nnx.List(
                RecurrentTransformerBlock(
                    config,
                    runtime,
                    row_relation=self.row_relation,
                    col_relation=self.col_relation,
                    box_relation=self.box_relation,
                    global_relation=self.global_relation,
                    rngs=rngs,
                )
                for _ in range(config.num_steps * config.layers_per_step)
            )

    def _build_communication_relations(
        self,
        row_ids: Array,
        col_ids: Array,
        box_ids: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Return Sudoku relations only for relation-based communication.

        Attention communication does not consume fixed relation matrices, so we
        keep those tensors zeroed rather than constructing unused Sudoku graph
        structure on that path.
        """
        if self.config.communication_type == "relation":
            return sudoku_relation_matrices(
                row_ids,
                col_ids,
                box_ids,
                seq_len=self.config.seq_len,
                include_global_relation=self.config.include_global_relation,
            )
        zero_relation = jnp.zeros((self.config.seq_len, self.config.seq_len), dtype=jnp.float32)
        return zero_relation, zero_relation, zero_relation, zero_relation

    def _token_entropy(self, hidden: Array) -> Array:
        logits = self.output_head(self.final_norm(hidden)).astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        probs = jnp.exp(log_probs)
        entropy = -jnp.sum(probs * log_probs, axis=-1, keepdims=True)
        normalizer = max(math.log(self.config.vocab_size), 1e-6)
        return jax.lax.stop_gradient(entropy / normalizer)

    def _blank_average(self, values: Array, given_mask: Array) -> Array:
        blank_mask = (~given_mask).astype(jnp.float32)[..., None]
        blank_normalizer = jnp.maximum(jnp.sum(blank_mask), 1.0)
        return jnp.sum(values * blank_mask) / blank_normalizer

    def _maybe_reinject_input(self, hidden: Array, input_hidden: Array) -> Array:
        if self.config.reinject_input:
            return hidden + input_hidden
        return hidden

    def _maybe_truncate_inner(self, hidden: Array, inner_index: int, *, train: bool) -> Array:
        if not train:
            return hidden
        first_grad_inner = self.config.inner_steps - self.config.grad_inner_steps
        if inner_index < first_grad_inner:
            return jax.lax.stop_gradient(hidden)
        return hidden

    @staticmethod
    def _fold_dropout_keys(step_keys: Array, inner_index: int, layer_index: int) -> tuple[Array, Array]:
        return (
            jax.random.fold_in(jax.random.fold_in(step_keys[0], inner_index), layer_index),
            jax.random.fold_in(jax.random.fold_in(step_keys[1], inner_index), layer_index),
        )

    def _run_universal_compute(
        self,
        hidden: Array,
        input_hidden: Array,
        step_embedding: Array,
        cell_type_embedding: Array,
        given_mask: Array,
        step_keys: Array,
        *,
        train: bool,
    ) -> tuple[Array, Array, Array]:
        current = hidden
        rho = jnp.ones((*hidden.shape[:2], 1), dtype=jnp.float32)
        alpha = jnp.ones((*hidden.shape[:2], 1), dtype=jnp.float32)
        for inner_index in range(self.config.inner_steps):
            for layer_index, block in enumerate(self.blocks):
                layer_input = self._maybe_reinject_input(current, input_hidden)
                communication_key, mlp_key = self._fold_dropout_keys(step_keys, inner_index, layer_index)
                updated, rho, alpha = block(
                    layer_input,
                    step_embedding,
                    cell_type_embedding,
                    given_mask,
                    self._token_entropy(layer_input) if self.config.uses_damped_transition else None,
                    train=train,
                    communication_dropout_key=communication_key if train else None,
                    mlp_dropout_key=mlp_key if train else None,
                )
                if self.config.freeze_clue_state:
                    updated = jnp.where(given_mask[..., None], current, updated)
                current = updated
            current = self._maybe_truncate_inner(current, inner_index, train=train)
        return current, rho, alpha

    def _run_recurrent_compute(
        self,
        hidden: Array,
        initial_hidden: Array,
        step_index: int,
        step_embedding: Array,
        given_mask: Array,
        step_keys: Array,
        *,
        train: bool,
    ) -> Array:
        current = hidden
        for inner_index in range(self.config.inner_steps):
            for layer_index in range(self.config.layers_per_step):
                block_index = step_index * self.config.layers_per_step + layer_index
                layer_input = self._maybe_reinject_input(current, initial_hidden)
                communication_key, mlp_key = self._fold_dropout_keys(step_keys, inner_index, layer_index)
                updated = self.blocks[block_index](
                    layer_input,
                    initial_hidden,
                    step_embedding,
                    train=train,
                    communication_dropout_key=communication_key if train else None,
                    mlp_dropout_key=mlp_key if train else None,
                )
                if self.config.freeze_clue_state:
                    updated = jnp.where(given_mask[..., None], current, updated)
                current = updated
            current = self._maybe_truncate_inner(current, inner_index, train=train)
        return current

    def _universal_transformer_recurrence(
        self,
        hidden: Array,
        cell_type_embedding: Array,
        given_mask: Array,
        step_embeddings: Array,
        step_dropout_keys: Array,
        *,
        train: bool,
    ) -> tuple[Array, Array, dict[str, Array]]:
        if self.config.uses_damped_transition:
            def scan_step(carry: Array, xs: tuple[Array, Array, Array]) -> tuple[Array, tuple[Array, Array, Array, Array]]:
                _, step_embedding, step_keys = xs
                updated, rho, alpha = self._run_universal_compute(
                    carry,
                    hidden,
                    step_embedding,
                    cell_type_embedding,
                    given_mask,
                    step_keys,
                    train=train,
                )
                hidden_delta = jnp.linalg.norm((updated - carry).astype(jnp.float32), axis=-1, keepdims=True)
                return updated, (
                    updated,
                    self._blank_average(rho, given_mask),
                    self._blank_average(alpha, given_mask),
                    self._blank_average(hidden_delta, given_mask),
                )

            hidden, (step_hidden, rho_mean, alpha_mean, hidden_delta_mean) = jax.lax.scan(
                scan_step,
                hidden,
                (self.step_ids, step_embeddings, step_dropout_keys),
            )
            return hidden, step_hidden, {
                "hidden_delta_mean": hidden_delta_mean,
                "rho_mean": rho_mean,
                "alpha_mean": alpha_mean,
            }

        def scan_step(carry: Array, xs: tuple[Array, Array, Array]) -> tuple[Array, tuple[Array, Array]]:
            _, step_embedding, step_keys = xs
            updated, _, _ = self._run_universal_compute(
                carry,
                hidden,
                step_embedding,
                cell_type_embedding,
                given_mask,
                step_keys,
                train=train,
            )
            hidden_delta = jnp.linalg.norm((updated - carry).astype(jnp.float32), axis=-1, keepdims=True)
            return updated, (updated, self._blank_average(hidden_delta, given_mask))

        hidden, (step_hidden, hidden_delta_mean) = jax.lax.scan(
            scan_step,
            hidden,
            (self.step_ids, step_embeddings, step_dropout_keys),
        )
        return hidden, step_hidden, {"hidden_delta_mean": hidden_delta_mean}

    def _recurrent_transformer_recurrence(
        self,
        hidden: Array,
        initial_hidden: Array,
        given_mask: Array,
        step_embeddings: Array,
        step_dropout_keys: Array,
        *,
        train: bool,
    ) -> tuple[Array, Array, dict[str, Array]]:
        step_hidden = []
        hidden_delta_mean = []
        carry = hidden
        for step_index in range(self.config.num_steps):
            step_keys = step_dropout_keys[step_index]
            updated = self._run_recurrent_compute(
                carry,
                initial_hidden,
                step_index,
                step_embeddings[step_index],
                given_mask,
                step_keys,
                train=train,
            )
            hidden_delta = jnp.linalg.norm((updated - carry).astype(jnp.float32), axis=-1, keepdims=True)
            carry = updated
            step_hidden.append(updated)
            hidden_delta_mean.append(self._blank_average(hidden_delta, given_mask))

        return carry, jnp.stack(step_hidden), {"hidden_delta_mean": jnp.stack(hidden_delta_mean)}

    def _prepare_recurrence_inputs(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> tuple[Array, Array, dict[str, Array], Array, Array]:
        if train:
            if dropout_key is None:
                raise ValueError("dropout_key must be provided when train=True")
            dropout_keys = jax.random.split(dropout_key, 1 + 2 * self.config.num_steps)
            embed_dropout_key = dropout_keys[0]
            step_dropout_keys = dropout_keys[1:].reshape(self.config.num_steps, 2)
        else:
            embed_dropout_key = None
            step_dropout_keys = self.inference_step_dropout_keys

        hidden, cell_type_embedding, given_mask = self.embeddings(tokens, train=train, dropout_key=embed_dropout_key)
        step_embeddings = self.step_embed(self.step_ids)
        if self.config.model_type == "universal_transformer":
            final_hidden, step_hidden, diagnostics = self._universal_transformer_recurrence(
                hidden,
                cell_type_embedding,
                given_mask,
                step_embeddings,
                step_dropout_keys,
                train=train,
            )
        else:
            final_hidden, step_hidden, diagnostics = self._recurrent_transformer_recurrence(
                hidden,
                hidden,
                given_mask,
                step_embeddings,
                step_dropout_keys,
                train=train,
            )
        return final_hidden, step_hidden, diagnostics, cell_type_embedding, given_mask

    def _logits_from_hidden(self, hidden: Array) -> Array:
        return self.output_head(self.final_norm(hidden)).astype(jnp.float32)

    def forward_all_steps(self, tokens: Array, *, train: bool, dropout_key: Array | None = None) -> Array:
        _, step_hidden, _, _, _ = self._prepare_recurrence_inputs(tokens, train=train, dropout_key=dropout_key)
        flat_hidden = step_hidden.reshape((-1, self.config.seq_len, self.config.d_model))
        flat_logits = self._logits_from_hidden(flat_hidden)
        return flat_logits.reshape(self.config.num_steps, tokens.shape[0], self.config.seq_len, self.config.vocab_size)

    def forward_all_steps_with_diagnostics(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        _, step_hidden, diagnostics, _, _ = self._prepare_recurrence_inputs(tokens, train=train, dropout_key=dropout_key)
        flat_hidden = step_hidden.reshape((-1, self.config.seq_len, self.config.d_model))
        flat_logits = self._logits_from_hidden(flat_hidden)
        step_logits = flat_logits.reshape(self.config.num_steps, tokens.shape[0], self.config.seq_len, self.config.vocab_size)
        return step_logits, diagnostics

    def __call__(self, tokens: Array, *, train: bool, dropout_key: Array | None = None) -> Array:
        hidden, _, _, _, _ = self._prepare_recurrence_inputs(tokens, train=train, dropout_key=dropout_key)
        return self._logits_from_hidden(hidden)
