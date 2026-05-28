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


def outputs_are_probabilities(model: object) -> bool:
    return (
        getattr(getattr(model, "config", None), "model_type", None) == "bdr"
        and getattr(getattr(model, "bdr", None), "update_rule", None) == "energy"
    )


def output_probabilities(model: object, outputs: jax.Array) -> jax.Array:
    if outputs_are_probabilities(model):
        return outputs.astype(jnp.float32)
    loss_type = getattr(getattr(model, "config", None), "loss_type", "softmax")
    if loss_type == "softmax":
        return jax.nn.softmax(outputs, axis=-1)
    elif loss_type == "stablemax":
        return stablemax(outputs, axis=-1)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def token_cross_entropy(model: object, outputs: jax.Array, targets: jax.Array) -> jax.Array:
    if outputs_are_probabilities(model):
        probs = outputs.astype(jnp.float32)
        target_probs = jnp.take_along_axis(probs, targets[..., None], axis=-1).squeeze(-1)
        return -jnp.log(target_probs + 1e-9)
    loss_type = getattr(getattr(model, "config", None), "loss_type", "softmax")
    if loss_type == "softmax":
        return optax.softmax_cross_entropy_with_integer_labels(outputs, targets)
    if loss_type == "stablemax":
        return stablemax_cross_entropy_with_integer_labels(outputs, targets)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def target_probability(model: object, outputs: jax.Array, targets: jax.Array) -> jax.Array:
    probs = output_probabilities(model, outputs)
    return jnp.take_along_axis(probs, targets[..., None], axis=-1).squeeze(-1)


def supervised_loss_mask(model: object, targets: jax.Array) -> jax.Array:
    if getattr(getattr(model, "config", None), "task_type", None) == "sudoku":
        return jnp.ones_like(targets, dtype=jnp.float32)
    # ARC-style datasets use label 0 as padding / ignore_label_id. Sudoku labels
    # use 0 for digit 1 and are fully supervised, so they are handled above.
    return (targets != 0).astype(jnp.float32)
