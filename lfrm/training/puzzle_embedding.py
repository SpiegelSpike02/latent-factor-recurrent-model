from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from lfrm.models.common import gather_embedding_rows


def act_sparse_puzzle_ids(
    carry: dict[str, jax.Array],
    batch: dict[str, jax.Array],
) -> jax.Array:
    return jnp.where(carry["halted"], batch["puzzle_identifiers"], carry["current_puzzle_identifiers"])


def sparse_puzzle_embeddings(model, puzzle_ids: jax.Array) -> jax.Array:
    return jax.lax.stop_gradient(model.puzzle_embed(puzzle_ids, train=True))


def update_sparse_puzzle_embeddings(
    model,
    puzzle_ids: jax.Array,
    puzzle_embedding_grads: jax.Array,
    *,
    learning_rate: jax.Array,
    weight_decay: float,
    coalesce_updates: bool,
) -> jax.Array:
    ids = puzzle_ids.reshape(-1).astype(jnp.int32)
    grads = puzzle_embedding_grads.reshape((ids.shape[0], puzzle_embedding_grads.shape[-1])).astype(jnp.float32)
    is_data_sharded = "data" in jax.sharding.get_abstract_mesh().axis_names
    if is_data_sharded:
        # The embedding table is replicated, so every replica must apply the
        # same sparse rows. Replicating this small update batch is cleaner than
        # sorting/coalescing over a sharded data axis.
        ids = jax.sharding.reshard(ids, P(None))
        grads = jax.sharding.reshard(grads, P(None, None))
    weights = model.puzzle_embed.weights[...]
    lr = learning_rate.astype(jnp.float32)
    if not coalesce_updates:
        # This path is only safe for small embedding tables. Large ARC puzzle
        # tables can make SPMD scatter choose very large temporary buffers.
        old_rows = gather_embedding_rows(weights, ids)
        new_rows = old_rows.astype(jnp.float32) * (1.0 - lr * weight_decay)
        new_rows = new_rows - lr * jnp.sign(grads)
        model.puzzle_embed.weights[...] = weights.at[ids].add(new_rows.astype(weights.dtype) - old_rows)
        return jnp.asarray(ids.shape[0], dtype=jnp.float32)

    unique_ids, inverse = jnp.unique(
        ids,
        return_inverse=True,
        size=ids.shape[0],
        fill_value=0,
    )
    grad_sums = jnp.zeros(
        (ids.shape[0], grads.shape[-1]),
        dtype=jnp.float32,
    ).at[inverse].add(grads)
    counts = jnp.zeros((ids.shape[0],), dtype=jnp.int32).at[inverse].add(1)
    valid = counts > 0

    old_rows = gather_embedding_rows(weights, unique_ids)
    new_rows = old_rows.astype(jnp.float32) * (1.0 - lr * weight_decay)
    new_rows = new_rows - lr * jnp.sign(grad_sums)
    row_delta = jnp.where(valid[:, None], new_rows.astype(weights.dtype) - old_rows, jnp.zeros_like(old_rows))
    model.puzzle_embed.weights[...] = weights.at[unique_ids].add(row_delta)
    return jnp.sum(valid).astype(jnp.float32)
