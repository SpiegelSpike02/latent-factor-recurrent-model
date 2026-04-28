from __future__ import annotations

import jax.numpy as jnp
import jax


Array = jax.Array


def compute_dtype(dtype_name: str) -> jnp.dtype:
    if dtype_name == "bfloat16":
        return jnp.bfloat16
    if dtype_name == "float32":
        return jnp.float32
    raise ValueError(f"Unsupported compute dtype: {dtype_name}")


def maybe_cast(x: Array, dtype: jnp.dtype) -> Array:
    return x if x.dtype == dtype else x.astype(dtype)
