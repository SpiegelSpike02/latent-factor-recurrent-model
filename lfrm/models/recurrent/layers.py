from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from lfrm.models.common import (
    Array,
    CastedEmbedding,
    casted_linear_init,
    compute_dtype,
    maybe_cast,
    trunc_normal,
    trunc_normal_init,
)


def rms_norm(x: Array, eps: float) -> Array:
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    return (x_f32 * jax.lax.rsqrt(variance + eps)).astype(x.dtype)


def rotate_half(x: Array) -> Array:
    first, second = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    *,
    bias: Array | None = None,
) -> Array:
    """Scaled dot-product attention in JAX's fused-friendly layout.

    Inputs and output use the public JAX SDPA layout `(batch, seq, heads,
    head_dim)`. JAX chooses the best supported implementation for the active
    backend, falling back to XLA when cuDNN fused attention is not applicable.
    """
    attention_bias = None if bias is None else bias[None, :, :, :].astype(jnp.float32)
    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        bias=attention_bias,
        implementation=None,
    )


def swiglu_intermediate_size(hidden_size: int, expansion: float, *, min_size: int = 1) -> int:
    raw = round(expansion * hidden_size * 2 / 3)
    return max(min_size, math.ceil(raw / 256) * 256)


__all__ = [
    "Array",
    "CastedEmbedding",
    "casted_linear_init",
    "compute_dtype",
    "dot_product_attention",
    "maybe_cast",
    "rms_norm",
    "rotate_half",
    "swiglu_intermediate_size",
    "trunc_normal",
    "trunc_normal_init",
]
