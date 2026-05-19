from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp
from flax import nnx

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


def apply_rope(q: Array, k: Array, cos: Array, sin: Array) -> tuple[Array, Array]:
    cos = cos[None, : q.shape[1], None, :].astype(jnp.float32)
    sin = sin[None, : q.shape[1], None, :].astype(jnp.float32)
    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    return (
        (q_f32 * cos + rotate_half(q_f32) * sin).astype(q.dtype),
        (k_f32 * cos + rotate_half(k_f32) * sin).astype(k.dtype),
    )


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


class SwiGLU(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        dtype: jnp.dtype,
        *,
        min_intermediate_size: int = 1,
        rngs: nnx.Rngs,
    ) -> None:
        intermediate_size = swiglu_intermediate_size(
            hidden_size,
            expansion,
            min_size=min_intermediate_size,
        )
        self.dtype = dtype
        self.gate_up = nnx.Linear(
            hidden_size,
            2 * intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.down = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        gate, up = jnp.split(self.gate_up(maybe_cast(x, self.dtype)), 2, axis=-1)
        return self.down(jax.nn.silu(gate) * up)


class FullAttention(nnx.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dtype: jnp.dtype,
        *,
        name: str,
        rngs: nnx.Rngs,
    ) -> None:
        if d_model % num_heads != 0:
            raise ValueError(f"{name} d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dtype = dtype
        self.qkv = nnx.Linear(
            d_model,
            3 * d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )
        self.out = nnx.Linear(
            d_model,
            d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            kernel_init=casted_linear_init,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Array,
        *,
        rope_cos: Array | None = None,
        rope_sin: Array | None = None,
        bias: Array | None = None,
    ) -> Array:
        batch_size, seq_len, d_model = hidden_states.shape
        qkv = self.qkv(maybe_cast(hidden_states, self.dtype))
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rope(q, k, rope_cos, rope_sin)
        attended = dot_product_attention(q, k, v, bias=bias)
        attended = attended.reshape(batch_size, seq_len, d_model)
        return self.out(attended)


__all__ = [
    "Array",
    "CastedEmbedding",
    "FullAttention",
    "SwiGLU",
    "apply_rope",
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
