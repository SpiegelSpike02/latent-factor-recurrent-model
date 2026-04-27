from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from config import ModelConfig, RuntimeConfig
from .common import Array, SwiGLU, compute_dtype
from .communication import build_communication_module
from .transitions import build_transition


class UniversalTransformerBlock(nnx.Module):
    """Universal Transformer block with pluggable communication and transition."""

    def __init__(
        self,
        config: ModelConfig,
        runtime: RuntimeConfig,
        *,
        row_relation: Array,
        col_relation: Array,
        box_relation: Array,
        global_relation: Array,
        rngs: nnx.Rngs,
    ) -> None:
        dtype = compute_dtype(runtime.compute_dtype)
        self.communication_norm = nnx.RMSNorm(config.d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.communication = build_communication_module(
            config,
            runtime,
            row_relation=row_relation,
            col_relation=col_relation,
            box_relation=box_relation,
            global_relation=global_relation,
            rngs=rngs,
        )
        self.transition = build_transition(config, runtime, rngs=rngs)
        self.ffn_norm = nnx.RMSNorm(config.d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.ffn = SwiGLU(config.d_model, config.d_ff, dropout_rate=config.dropout_rate, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden: Array,
        step_embedding: Array,
        cell_type_embedding: Array,
        given_mask: Array,
        entropy: Array | None,
        *,
        train: bool,
        communication_dropout_key: Array | None,
        mlp_dropout_key: Array | None,
    ) -> tuple[Array, Array, Array]:
        step_condition = step_embedding[None, None, :]
        # Step information conditions the computation, but is not written into
        # the persistent state by itself. This keeps hidden deltas meaningful as
        # a convergence signal: zero communication/FFN update means zero change.
        communication_input = self.communication_norm(hidden) + step_condition
        communication_delta = self.communication(
            communication_input,
            train=train,
            dropout_key=communication_dropout_key,
        )
        rho, alpha = self.transition.scales(
            state=hidden,
            step_embedding=step_embedding,
            cell_type_embedding=cell_type_embedding,
            entropy=entropy,
            given_mask=given_mask,
            batch_shape=hidden.shape[:2],
        )
        updated_context = hidden + rho.astype(communication_delta.dtype) * communication_delta
        ffn_input = self.ffn_norm(updated_context) + step_condition
        candidate = updated_context + self.ffn(ffn_input, train=train, dropout_key=mlp_dropout_key)
        return self.transition.apply(hidden, candidate, alpha), rho, alpha
