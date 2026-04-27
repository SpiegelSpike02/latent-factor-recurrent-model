from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from utils.relations import normalized_relation_matrix


def sudoku_box_ids(row_ids: jax.Array, col_ids: jax.Array, *, grid_height: int, grid_width: int) -> jax.Array:
    box_height = int(round(math.sqrt(grid_height)))
    box_width = int(round(math.sqrt(grid_width)))
    if box_height * box_height != grid_height:
        return jnp.arange(grid_height * grid_width, dtype=jnp.int32)
    if box_width * box_width != grid_width:
        return jnp.arange(grid_height * grid_width, dtype=jnp.int32)
    return (row_ids // box_height) * box_width + (col_ids // box_width)


def sudoku_num_box_units(*, grid_height: int, grid_width: int, seq_len: int) -> int:
    box_height = int(round(math.sqrt(grid_height)))
    box_width = int(round(math.sqrt(grid_width)))
    if box_height * box_height != grid_height:
        return seq_len
    if box_width * box_width != grid_width:
        return seq_len
    return (grid_height // box_height) * (grid_width // box_width)


def sudoku_relation_matrices(
    row_ids: jax.Array,
    col_ids: jax.Array,
    box_ids: jax.Array,
    *,
    seq_len: int,
    include_global_relation: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    row_relation = normalized_relation_matrix(row_ids[:, None] == row_ids[None, :])
    col_relation = normalized_relation_matrix(col_ids[:, None] == col_ids[None, :])
    box_relation = normalized_relation_matrix(box_ids[:, None] == box_ids[None, :])
    if include_global_relation:
        global_relation = normalized_relation_matrix(jnp.ones((seq_len, seq_len), dtype=bool))
    else:
        global_relation = jnp.zeros((seq_len, seq_len), dtype=jnp.float32)
    return row_relation, col_relation, box_relation, global_relation


def sudoku_unit_matrices(
    row_ids: jax.Array,
    col_ids: jax.Array,
    box_ids: jax.Array,
    *,
    grid_height: int,
    grid_width: int,
    num_box_units: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    row_unit_matrix = jax.nn.one_hot(row_ids, grid_height, dtype=jnp.float32).T
    col_unit_matrix = jax.nn.one_hot(col_ids, grid_width, dtype=jnp.float32).T
    box_unit_matrix = jax.nn.one_hot(box_ids, num_box_units, dtype=jnp.float32).T
    return row_unit_matrix, col_unit_matrix, box_unit_matrix


def apply_given_logits(logits: jax.Array, inputs: jax.Array, given_mask: jax.Array) -> jax.Array:
    fixed_logits = -30.0 + jax.nn.one_hot(inputs, logits.shape[-1], dtype=logits.dtype) * 60.0
    return jnp.where(given_mask[..., None], fixed_logits, logits)


def apply_given_logits_by_step(
    step_logits: jax.Array,
    inputs: jax.Array,
    given_mask: jax.Array,
) -> jax.Array:
    return apply_given_logits(step_logits, inputs[None, :, :], given_mask[None, :, :])


def soft_sudoku_validity_loss(
    step_logits: jax.Array,
    row_unit_matrix: jax.Array,
    col_unit_matrix: jax.Array,
    box_unit_matrix: jax.Array,
) -> jax.Array:
    digit_probs = jax.nn.softmax(step_logits.astype(jnp.float32), axis=-1)[..., 2:11]
    row_counts = jnp.einsum("us,kbsd->kbud", row_unit_matrix, digit_probs)
    col_counts = jnp.einsum("us,kbsd->kbud", col_unit_matrix, digit_probs)
    box_counts = jnp.einsum("us,kbsd->kbud", box_unit_matrix, digit_probs)
    unit_counts = jnp.concatenate([row_counts, col_counts, box_counts], axis=2)
    return jnp.mean(jnp.square(unit_counts - 1.0), axis=(1, 2, 3))
