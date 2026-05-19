from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def stablemax(logits: jax.Array, axis: int = -1) -> jax.Array:
    negative_logits = jnp.where(logits < 0.0, logits, 0.0)
    positive = jnp.where(logits >= 0.0, logits + 1.0, 1.0 / (1.0 - negative_logits))
    return positive / jnp.sum(positive, axis=axis, keepdims=True)


def stablemax_cross_entropy_with_integer_labels(logits: jax.Array, targets: jax.Array) -> jax.Array:
    positive_logits = jnp.where(logits >= 0.0, logits, 0.0)
    negative_logits = jnp.where(logits < 0.0, logits, 0.0)
    log_positive = jnp.where(
        logits >= 0.0,
        jnp.log1p(positive_logits),
        -jnp.log1p(-negative_logits),
    )
    log_normalizer = jax.nn.logsumexp(log_positive, axis=-1)
    target_log_positive = jnp.take_along_axis(log_positive, targets[..., None], axis=-1).squeeze(-1)
    return log_normalizer - target_log_positive


def token_cross_entropy(model: object, logits: jax.Array, targets: jax.Array) -> jax.Array:
    loss_type = getattr(getattr(model, "config", None), "loss_type", "softmax")
    if loss_type == "softmax":
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    if loss_type == "stablemax":
        return stablemax_cross_entropy_with_integer_labels(logits, targets)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def target_probability(model: object, logits: jax.Array, targets: jax.Array) -> jax.Array:
    loss_type = getattr(getattr(model, "config", None), "loss_type", "softmax")
    if loss_type == "softmax":
        probs = jax.nn.softmax(logits, axis=-1)
    elif loss_type == "stablemax":
        probs = stablemax(logits, axis=-1)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    return jnp.take_along_axis(probs, targets[..., None], axis=-1).squeeze(-1)


def masked_token_ce(model: object, logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
    mask_f32 = mask.astype(jnp.float32)
    normalizer = jnp.maximum(jnp.sum(mask_f32), 1.0)
    token_loss = token_cross_entropy(model, logits, targets)
    return jnp.sum(token_loss * mask_f32) / normalizer


def target_loss_weights(model: object, targets: jax.Array) -> jax.Array:
    if getattr(getattr(model, "config", None), "task_type", "sudoku") != "maze":
        return jnp.asarray(1.0, dtype=jnp.float32)
    path_weight = getattr(getattr(model, "config", None), "path_loss_weight", 1.0)
    if path_weight == 1.0:
        return jnp.asarray(1.0, dtype=jnp.float32)
    return jnp.where(targets == 5, jnp.asarray(path_weight, dtype=jnp.float32), 1.0)


def loss_mask_from_given(model: object, given_mask: jax.Array) -> jax.Array:
    supervision = getattr(getattr(model, "config", None), "supervision", "unknown_only")
    if supervision == "unknown_only":
        return (~given_mask).astype(jnp.float32)
    if supervision == "full_grid":
        return jnp.ones_like(given_mask, dtype=jnp.float32)
    raise ValueError(f"Unsupported supervision mode: {supervision}")


def weighted_mask_normalizer(mask: jax.Array, weights: jax.Array) -> jax.Array:
    return jnp.maximum(jnp.sum(mask.astype(jnp.float32) * weights.astype(jnp.float32)), 1.0)
