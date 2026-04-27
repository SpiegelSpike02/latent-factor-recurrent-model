from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from config import ModelConfig, RuntimeConfig
from .common import Array
from .damped_transition import DampedTransitionScales


class ResidualTransition(nnx.Module):
    """Standard residual transition: the candidate state becomes the next state."""

    def scales(
        self,
        *,
        batch_shape: tuple[int, int],
        state: Array | None = None,
        step_embedding: Array | None = None,
        cell_type_embedding: Array | None = None,
        entropy: Array | None = None,
        given_mask: Array | None = None,
    ) -> tuple[Array, Array]:
        del state, step_embedding, cell_type_embedding, entropy, given_mask
        ones = jnp.ones((*batch_shape, 1), dtype=jnp.float32)
        return ones, ones

    def apply(self, hidden: Array, candidate: Array, alpha: Array) -> Array:
        del hidden, alpha
        return candidate


class DampedTransition(nnx.Module):
    """Continuous damped transition with learned rho/alpha scales."""

    def __init__(self, config: ModelConfig, runtime: RuntimeConfig, *, rngs: nnx.Rngs) -> None:
        self.scale_predictor = DampedTransitionScales(config, runtime, rngs=rngs)

    def scales(
        self,
        *,
        state: Array,
        step_embedding: Array,
        cell_type_embedding: Array,
        entropy: Array | None,
        given_mask: Array,
        batch_shape: tuple[int, int] | None = None,
    ) -> tuple[Array, Array]:
        del batch_shape
        rho, alpha = self.scale_predictor(
            state,
            step_embedding,
            cell_type_embedding,
            entropy,
            given_mask=given_mask,
        )
        return rho, alpha

    def apply(self, hidden: Array, candidate: Array, alpha: Array) -> Array:
        return hidden + alpha.astype(candidate.dtype) * (candidate - hidden)


def build_transition(config: ModelConfig, runtime: RuntimeConfig, *, rngs: nnx.Rngs) -> nnx.Module:
    if config.uses_damped_transition:
        return DampedTransition(config, runtime, rngs=rngs)
    return ResidualTransition()
