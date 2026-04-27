from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from config import ModelConfig, RuntimeConfig
from .common import Array, SwiGLU, compute_dtype
from .communication import build_communication_module


class RecurrentTransformerBlock(nnx.Module):
    """One recurrent Transformer step with pluggable communication."""

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
        self.input_proj = nnx.Linear(2 * config.d_model, config.d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
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
        self.ffn_norm = nnx.RMSNorm(config.d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.ffn = SwiGLU(config.d_model, config.d_ff, dropout_rate=config.dropout_rate, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden: Array,
        initial_hidden: Array,
        step_embedding: Array,
        *,
        train: bool,
        communication_dropout_key: Array | None,
        mlp_dropout_key: Array | None,
    ) -> Array:
        step_condition = step_embedding[None, None, :]
        # The recurrent step can condition on both current and initial state,
        # while the step embedding modulates computation instead of becoming an
        # unconditional persistent-state increment.
        x = hidden + self.input_proj(jnp.concatenate([hidden, initial_hidden], axis=-1))
        communicated = x + self.communication(
            self.communication_norm(x) + step_condition,
            train=train,
            dropout_key=communication_dropout_key,
        )
        return communicated + self.ffn(
            self.ffn_norm(communicated) + step_condition,
            train=train,
            dropout_key=mlp_dropout_key,
        )
