from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from config import ModelConfig, RuntimeConfig
from .common import Array, compute_dtype, maybe_cast


class GlobalSelfAttention(nnx.Module):
    """Dense global self-attention communication."""

    def __init__(
        self,
        config: ModelConfig,
        runtime: RuntimeConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        dtype = compute_dtype(runtime.compute_dtype)
        self.dtype = dtype
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.qkv_proj = nnx.Linear(
            config.d_model,
            3 * config.d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            config.d_model,
            config.d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(self, hidden: Array, *, train: bool, dropout_key: Array | None) -> Array:
        hidden = maybe_cast(hidden, self.dtype)
        batch_size, seq_len, _ = hidden.shape
        qkv = self.qkv_proj(hidden)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.moveaxis(qkv, 2, 0)
        q = jnp.swapaxes(q, 1, 2)
        k = jnp.swapaxes(k, 1, 2)
        v = jnp.swapaxes(v, 1, 2)
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(self.head_dim)
        weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
        weights = self.dropout(weights, deterministic=not train, rngs=dropout_key)
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        attended = jnp.swapaxes(attended, 1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(attended)


class TypedRelationCommunication(nnx.Module):
    """Communication over fixed row/column/box/global relations.

    This module is intentionally independent from the recurrent update rule, so
    relation-vs-attention can be ablated separately from UT-vs-RT.
    """

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
        self.dtype = dtype
        self.include_global_relation = config.include_global_relation
        self.row_relation = row_relation.astype(jnp.float32)
        self.col_relation = col_relation.astype(jnp.float32)
        self.box_relation = box_relation.astype(jnp.float32)
        self.global_relation = global_relation.astype(jnp.float32)
        self.row_proj = nnx.Linear(config.d_model, config.d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.col_proj = nnx.Linear(config.d_model, config.d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.box_proj = nnx.Linear(config.d_model, config.d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        relation_count = 3
        if self.include_global_relation:
            self.global_proj = nnx.Linear(config.d_model, config.d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
            relation_count += 1
        self.mix_in = nnx.Linear(relation_count * config.d_model, config.d_model, use_bias=True, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.mix_out = nnx.Linear(config.d_model, config.d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        if self.include_global_relation:
            self.raw_global_strength = nnx.Param(jnp.asarray(0.0, dtype=jnp.float32))
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    @staticmethod
    def _aggregate(relation: Array, values: Array) -> Array:
        return jnp.einsum("nm,bmd->bnd", relation, values)

    def __call__(self, hidden: Array, *, train: bool, dropout_key: Array | None) -> Array:
        hidden = maybe_cast(hidden, self.dtype)
        row_message = self._aggregate(self.row_relation, self.row_proj(hidden))
        col_message = self._aggregate(self.col_relation, self.col_proj(hidden))
        box_message = self._aggregate(self.box_relation, self.box_proj(hidden))
        relation_messages = [row_message, col_message, box_message]
        if self.include_global_relation:
            global_message = self._aggregate(self.global_relation, self.global_proj(hidden))
            global_strength = 0.2 * jax.nn.sigmoid(jnp.asarray(self.raw_global_strength))
            relation_messages.append(global_message * global_strength.astype(global_message.dtype))
        relation_stack = jnp.concatenate(relation_messages, axis=-1)
        message_delta = self.mix_out(jax.nn.silu(self.mix_in(relation_stack)))
        return self.dropout(message_delta, deterministic=not train, rngs=dropout_key)


def build_communication_module(
    config: ModelConfig,
    runtime: RuntimeConfig,
    *,
    row_relation: Array,
    col_relation: Array,
    box_relation: Array,
    global_relation: Array,
    rngs: nnx.Rngs,
) -> nnx.Module:
    if config.communication_type == "relation":
        return TypedRelationCommunication(
            config,
            runtime,
            row_relation=row_relation,
            col_relation=col_relation,
            box_relation=box_relation,
            global_relation=global_relation,
            rngs=rngs,
        )
    if config.communication_type == "attention":
        return GlobalSelfAttention(config, runtime, rngs=rngs)
    raise ValueError(f"Unsupported communication_type: {config.communication_type}")
