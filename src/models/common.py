from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


Array = jax.Array


def compute_dtype(dtype_name: str) -> jnp.dtype:
    if dtype_name == "bfloat16":
        return jnp.bfloat16
    if dtype_name == "float32":
        return jnp.float32
    raise ValueError(f"Unsupported compute dtype: {dtype_name}")


def maybe_cast(x: Array, dtype: jnp.dtype) -> Array:
    return x if x.dtype == dtype else x.astype(dtype)


class SwiGLU(nnx.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        dropout_rate: float,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ) -> None:
        self.dtype = dtype
        self.in_proj = nnx.Linear(
            d_model,
            2 * d_ff,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            d_ff,
            d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: Array, *, train: bool, dropout_key: Array | None) -> Array:
        hidden = self.in_proj(maybe_cast(x, self.dtype))
        activation, value = jnp.split(hidden, 2, axis=-1)
        return self.dropout(
            self.out_proj(jax.nn.silu(activation) * value),
            deterministic=not train,
            rngs=dropout_key,
        )
