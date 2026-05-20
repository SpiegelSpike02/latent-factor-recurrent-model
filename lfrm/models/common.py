from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P


Array = jax.Array


def compute_dtype(dtype_name: str) -> jnp.dtype:
    if dtype_name == "bfloat16":
        return jnp.bfloat16
    if dtype_name == "float32":
        return jnp.float32
    raise ValueError(f"Unsupported compute dtype: {dtype_name}")


def maybe_cast(x: Array, dtype: jnp.dtype) -> Array:
    return x if x.dtype == dtype else x.astype(dtype)


def leading_axis_gather_out_sharding(indices: Array) -> NamedSharding | None:
    sharding = getattr(indices, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        return None
    index_spec = tuple(sharding.spec)
    if len(index_spec) < indices.ndim:
        index_spec = (*index_spec, *([None] * (indices.ndim - len(index_spec))))
    return NamedSharding(sharding.mesh, P(*index_spec, None))


def gather_embedding_rows(weights: Array, identifiers: Array) -> Array:
    identifiers = identifiers.astype(jnp.int32)
    out_sharding = leading_axis_gather_out_sharding(identifiers)
    if out_sharding is None:
        return weights.at[identifiers].get()
    return weights.at[identifiers].get(out_sharding=out_sharding)


def trunc_normal(key: Array, shape: tuple[int, ...], std: float, dtype: jnp.dtype = jnp.float32) -> Array:
    """JAX version of the official TRM truncated-normal initializer."""
    if std == 0.0:
        return jnp.zeros(shape, dtype=dtype)
    lower = -2.0
    upper = 2.0
    sqrt2 = math.sqrt(2.0)
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)
    z = (b - a) / 2.0
    c = (2.0 * math.pi) ** -0.5
    pdf_u = c * math.exp(-0.5 * lower**2)
    pdf_l = c * math.exp(-0.5 * upper**2)
    comp_std = std / math.sqrt(
        1.0
        - (upper * pdf_u - lower * pdf_l) / z
        - ((pdf_u - pdf_l) / z) ** 2
    )
    values = jax.random.uniform(key, shape, minval=a, maxval=b, dtype=jnp.float32)
    values = jax.lax.erf_inv(values) * (sqrt2 * comp_std)
    values = jnp.clip(values, lower * comp_std, upper * comp_std)
    return values.astype(dtype)


def trunc_normal_init(std: float):
    def init(key: Array, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32) -> Array:
        return trunc_normal(key, shape, std=std, dtype=dtype)

    return init


def casted_linear_init(key: Array, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32) -> Array:
    if len(shape) < 1:
        raise ValueError("Linear kernel shape must have at least one dimension")
    return trunc_normal(key, shape, std=1.0 / math.sqrt(shape[0]), dtype=dtype)


class CastedEmbedding(nnx.Module):
    """Puzzle embedding table stored in float32 and gathered in compute dtype."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: jnp.dtype,
        *,
        init_std: float = 0.0,
        rngs: nnx.Rngs,
    ) -> None:
        if num_embeddings < 1:
            raise ValueError("num_embeddings must be at least 1")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")
        weights = (
            jnp.zeros((num_embeddings, embedding_dim), dtype=jnp.float32)
            if init_std == 0.0
            else trunc_normal(rngs.params(), (num_embeddings, embedding_dim), init_std, jnp.float32)
        )
        self.weights = nnx.Param(weights)
        self.dtype = dtype

    def __call__(self, identifiers: Array, *, train: bool) -> Array:
        del train
        embedding = gather_embedding_rows(self.weights[...], identifiers)
        return maybe_cast(embedding, self.dtype)
