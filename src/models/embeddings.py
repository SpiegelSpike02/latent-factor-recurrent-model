from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from config import ModelConfig, RuntimeConfig
from .common import Array, compute_dtype


class GridEmbeddings(nnx.Module):
    """Token, position, box, and clue/blank embeddings for grid tasks."""

    def __init__(
        self,
        config: ModelConfig,
        runtime: RuntimeConfig,
        *,
        row_ids: Array,
        col_ids: Array,
        box_ids: Array,
        num_box_units: int,
        rngs: nnx.Rngs,
    ) -> None:
        dtype = compute_dtype(runtime.compute_dtype)
        self.config = config
        self.row_ids = row_ids
        self.col_ids = col_ids
        self.box_ids = box_ids
        self.token_embed = nnx.Embed(config.vocab_size, config.d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.row_embed = nnx.Embed(config.grid_height, config.d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.col_embed = nnx.Embed(config.grid_width, config.d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.box_embed = nnx.Embed(num_box_units, config.d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.cell_type_embed = nnx.Embed(2, config.d_model, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(
        self,
        tokens: Array,
        *,
        train: bool,
        dropout_key: Array | None,
    ) -> tuple[Array, Array, Array]:
        hidden = self.token_embed(tokens)
        hidden = hidden + self.row_embed(self.row_ids)[None, :, :]
        hidden = hidden + self.col_embed(self.col_ids)[None, :, :]
        hidden = hidden + self.box_embed(self.box_ids)[None, :, :]
        given_mask = tokens != 1
        cell_type_embedding = self.cell_type_embed(given_mask.astype(jnp.int32))
        if self.config.use_clue_type_embedding:
            hidden = hidden + cell_type_embedding
        else:
            cell_type_embedding = jnp.zeros_like(cell_type_embedding)
        hidden = self.dropout(hidden, deterministic=not train, rngs=dropout_key)
        return hidden, cell_type_embedding, given_mask
