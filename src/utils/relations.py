from __future__ import annotations

import jax
import jax.numpy as jnp


def normalized_relation_matrix(mask: jax.Array) -> jax.Array:
    """Build a row-normalized relation matrix with self edges removed."""
    mask_f = mask.astype(jnp.float32)
    eye = jnp.eye(mask.shape[0], dtype=jnp.float32)
    mask_wo_self = mask_f * (1.0 - eye)
    normalizer = jnp.maximum(jnp.sum(mask_wo_self, axis=-1, keepdims=True), 1.0)
    return mask_wo_self / normalizer
