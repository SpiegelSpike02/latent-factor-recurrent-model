from __future__ import annotations

import math
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


def unscaled_rms_norm(num_features: int, eps: float, dtype: jnp.dtype, rngs: nnx.Rngs) -> nnx.RMSNorm:
    return nnx.RMSNorm(
        num_features,
        epsilon=eps,
        dtype=dtype,
        param_dtype=jnp.float32,
        use_scale=False,
        rngs=rngs,
    )


def rotate_half(x: Array) -> Array:
    first, second = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def apply_rope(q: Array, k: Array, cos: Array, sin: Array) -> tuple[Array, Array]:
    cos = cos[None, : q.shape[1], None, :].astype(q.dtype)
    sin = sin[None, : q.shape[1], None, :].astype(q.dtype)
    return (
        q * cos + rotate_half(q) * sin,
        k * cos + rotate_half(k) * sin,
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
    head_dim)`. We pass `implementation=None` so JAX/XLA chooses the backend.
    """
    attention_bias = None if bias is None else bias[None, :, :, :].astype(jnp.float32)
    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        bias=attention_bias,
        implementation=None,
    )


def multi_head_attention_with_rope(
    attention: nnx.MultiHeadAttention,
    query_input: Array,
    key_input: Array,
    value_input: Array,
    *,
    num_heads: int,
    head_dim: int,
    dtype: jnp.dtype,
    rope_cos: Array | None = None,
    rope_sin: Array | None = None,
    bias: Array | None = None,
    name: str = "attention",
) -> Array:
    q = attention.query(maybe_cast(query_input, dtype))
    k = attention.key(maybe_cast(key_input, dtype))
    v = attention.value(maybe_cast(value_input, dtype))
    expected_heads = (num_heads, head_dim)
    if q.shape[-2:] != expected_heads or k.shape[-2:] != expected_heads or v.shape[-2:] != expected_heads:
        raise ValueError(f"{name} projection shape does not match configured heads")
    if rope_cos is not None and rope_sin is not None:
        q, k = apply_rope(q, k, rope_cos, rope_sin)
    attended = dot_product_attention(q, k, v, bias=bias)
    return attention.out(attended)


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
        self.name = name
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dtype = dtype
        self.attention = nnx.MultiHeadAttention(
            num_heads,
            in_features=d_model,
            qkv_features=d_model,
            out_features=d_model,
            dtype=dtype,
            param_dtype=jnp.float32,
            use_bias=False,
            dropout_rate=0.0,
            attention_fn=dot_product_attention,
            kernel_init=casted_linear_init,
            out_kernel_init=casted_linear_init,
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
        return multi_head_attention_with_rope(
            self.attention,
            hidden_states,
            hidden_states,
            hidden_states,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            bias=bias,
            name=self.name,
        )


class ConvSwiGLU1D(nnx.Module):
    """Sequence depthwise-conv SwiGLU block used by recurrent grid models."""

    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        kernel_size: int,
        dtype: jnp.dtype,
        *,
        min_intermediate_size: int = 256,
        use_bias: bool = True,
        rngs: nnx.Rngs,
    ) -> None:
        if kernel_size < 1:
            raise ValueError("ConvSwiGLU1D kernel_size must be positive")
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
        self.depthwise = nnx.Conv(
            intermediate_size,
            intermediate_size,
            kernel_size=(kernel_size,),
            padding=[(kernel_size // 2, kernel_size // 2)],
            feature_group_count=intermediate_size,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=jnp.float32,
            preferred_element_type=dtype,
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

    def __call__(self, h: Array) -> Array:
        gate, up = jnp.split(self.gate_up(maybe_cast(h, self.dtype)), 2, axis=-1)
        x = jax.nn.silu(gate) * up
        x = self.depthwise(x)[:, : h.shape[1], :]
        return self.down(jax.nn.silu(x))


class ConvSwiGLU2D(nnx.Module):
    """Grid depthwise-conv SwiGLU block for fixed-height 2D token grids."""

    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        kernel_size: int,
        grid_height: int,
        grid_width: int,
        dtype: jnp.dtype,
        *,
        min_intermediate_size: int = 1,
        use_bias: bool = True,
        rngs: nnx.Rngs,
    ) -> None:
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("ConvSwiGLU2D kernel_size must be a positive odd integer")
        intermediate_size = swiglu_intermediate_size(
            hidden_size,
            expansion,
            min_size=min_intermediate_size,
        )
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
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
        self.depthwise = nnx.Conv(
            intermediate_size,
            intermediate_size,
            kernel_size=(kernel_size, kernel_size),
            padding="SAME",
            feature_group_count=intermediate_size,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=jnp.float32,
            preferred_element_type=dtype,
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

    def __call__(self, h: Array) -> Array:
        batch_size = h.shape[0]
        gate, up = jnp.split(self.gate_up(maybe_cast(h, self.dtype)), 2, axis=-1)
        x = jax.nn.silu(gate) * up
        x = x.reshape(batch_size, self.grid_height, self.grid_width, x.shape[-1])
        x = self.depthwise(x)
        x = x.reshape(batch_size, self.grid_height * self.grid_width, x.shape[-1])
        return self.down(jax.nn.silu(x))


def build_1d_rope(num_positions: int, head_dim: int, theta: float) -> tuple[Array, Array]:
    if head_dim % 2 != 0:
        raise ValueError("1D RoPE head dimension must be even")
    positions = jnp.arange(num_positions, dtype=jnp.float32)
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = jnp.einsum("n,d->nd", positions, inv_freq)
    rope = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(rope), jnp.sin(rope)


def build_2d_axial_rope(
    rows: Array,
    cols: Array,
    head_dim: int,
    theta: float,
) -> tuple[Array, Array]:
    if head_dim % 4 != 0:
        raise ValueError("2D axial RoPE head dimension must be divisible by 4")
    axis_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, axis_dim, 2, dtype=jnp.float32) / axis_dim))
    row_freqs = rows.astype(jnp.float32)[:, None] * inv_freq[None, :]
    col_freqs = cols.astype(jnp.float32)[:, None] * inv_freq[None, :]
    freqs = jnp.concatenate((row_freqs, col_freqs), axis=-1)
    rope = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(rope), jnp.sin(rope)


def run_truncated_h_cycles(
    initial_state: Array,
    *,
    h_cycles: int,
    latent_update,
) -> Array:
    """Run H cycles with gradients only through the final latent update."""
    if h_cycles < 1:
        raise ValueError("h_cycles must be at least 1")

    def h_cycle_body(_: int, state: Array) -> Array:
        return jax.lax.stop_gradient(latent_update(state))

    state = jax.lax.fori_loop(0, h_cycles - 1, h_cycle_body, initial_state)
    return latent_update(state)


__all__ = [
    "Array",
    "CastedEmbedding",
    "ConvSwiGLU1D",
    "ConvSwiGLU2D",
    "FullAttention",
    "SwiGLU",
    "apply_rope",
    "build_1d_rope",
    "build_2d_axial_rope",
    "casted_linear_init",
    "compute_dtype",
    "dot_product_attention",
    "maybe_cast",
    "multi_head_attention_with_rope",
    "rms_norm",
    "rotate_half",
    "run_truncated_h_cycles",
    "swiglu_intermediate_size",
    "trunc_normal",
    "trunc_normal_init",
    "unscaled_rms_norm",
]
