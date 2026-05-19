from __future__ import annotations

import math
import os

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
    head_dim)`. Attention defaults to cuDNN fused attention because XLA's auto
    choice can still lower to explicit dot/softmax on some JAX/CUDA stacks. Set
    `LFRM_ATTENTION_IMPLEMENTATION=auto|cudnn|xla` to override this selection.
    """
    attention_bias = None if bias is None else bias[None, :, :, :].astype(jnp.float32)
    requested = os.environ.get("LFRM_ATTENTION_IMPLEMENTATION", "cudnn").strip().lower()
    implementation: str | None
    if requested in ("", "auto", "none"):
        implementation = "cudnn" if attention_bias is None and jax.default_backend() == "gpu" else None
    elif requested in ("cudnn", "xla"):
        implementation = requested
    else:
        raise ValueError(
            "LFRM_ATTENTION_IMPLEMENTATION must be one of auto, cudnn, or xla; "
            f"got {requested!r}"
        )
    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        bias=attention_bias,
        implementation=implementation,
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
