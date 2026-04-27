from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from config import ModelConfig, RuntimeConfig
from .common import Array, compute_dtype


class DampedTransitionScales(nnx.Module):
    """Predicts continuous read scaling and damping coefficients."""

    def __init__(self, config: ModelConfig, runtime: RuntimeConfig, *, rngs: nnx.Rngs) -> None:
        dtype = compute_dtype(runtime.compute_dtype)
        self.freeze_clue_state = config.freeze_clue_state
        self.state_norm = nnx.RMSNorm(config.d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        hidden_dim = config.transition_hidden_dim
        self.state_proj = nnx.Linear(config.d_model, hidden_dim, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.step_proj = nnx.Linear(config.d_model, hidden_dim, use_bias=False, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.cell_type_proj = nnx.Linear(config.d_model, hidden_dim, use_bias=True, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.entropy_proj = nnx.Linear(1, hidden_dim, use_bias=True, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.rho_head = nnx.Linear(hidden_dim, 1, use_bias=True, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.alpha_head = nnx.Linear(hidden_dim, 1, use_bias=True, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

    def __call__(
        self,
        state: Array,
        step_embedding: Array,
        cell_type_embedding: Array,
        entropy: Array | None,
        given_mask: Array,
    ) -> tuple[Array, Array]:
        features = (
            self.state_proj(self.state_norm(state))
            + self.step_proj(step_embedding)[None, None, :]
            + self.cell_type_proj(cell_type_embedding)
        )
        if entropy is not None:
            features = features + self.entropy_proj(entropy)
        features = jax.nn.silu(features)
        rho = jax.nn.sigmoid(self.rho_head(features).astype(jnp.float32))
        alpha = jax.nn.sigmoid(self.alpha_head(features).astype(jnp.float32))
        if self.freeze_clue_state:
            alpha = alpha * (~given_mask).astype(jnp.float32)[..., None]
        return rho, alpha
