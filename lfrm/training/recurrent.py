from __future__ import annotations

import jax
from flax import nnx

from lfrm.models import TinyRecursiveModel
from lfrm.training.optim import scheduled_lr, trainable_param_filter, uses_sparse_puzzle_embedding
from lfrm.training.puzzle_embedding import (
    act_sparse_puzzle_ids,
    sparse_puzzle_embeddings,
    update_sparse_puzzle_embeddings,
)
from lfrm.training.steps import (
    trm_act_loss_and_metrics,
    trm_dense_unroll_loss_and_metrics,
    trm_eval_loss_and_metrics,
)


def build_trm_act_train_step_runner(config, halt_loss_weight: float = 0.5):
    use_sparse_puzzle_embed = uses_sparse_puzzle_embedding(config)
    puzzle_lr_schedule = scheduled_lr(
        peak_value=config.optimizer.puzzle_embed_learning_rate,
        min_ratio=config.optimizer.lr_min_ratio,
        warmup_steps=max(1, config.optimizer.lr_warmup_steps),
        optimizer_updates=max(1, config.train.optimizer_updates),
    )
    diff_filter = trainable_param_filter(config)

    def train_step(
        model: TinyRecursiveModel,
        optimizer: nnx.Optimizer,
        carry: dict[str, jax.Array],
        batch: dict[str, jax.Array],
        dropout_key: jax.Array,
        optimizer_step: jax.Array,
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        dropout_key = jax.random.fold_in(dropout_key, optimizer_step)
        puzzle_ids = act_sparse_puzzle_ids(carry, batch)
        puzzle_embeddings = (
            sparse_puzzle_embeddings(model, puzzle_ids)
            if use_sparse_puzzle_embed
            else None
        )

        def objective(model, puzzle_embeddings, carry, batch, train, dropout_key):
            return trm_act_loss_and_metrics(
                model,
                carry,
                batch,
                train,
                dropout_key,
                halt_loss_weight,
                puzzle_embeddings=puzzle_embeddings,
            )

        grad_argnums = (
            (nnx.DiffState(0, diff_filter), 1)
            if use_sparse_puzzle_embed
            else nnx.DiffState(0, diff_filter)
        )
        (_, (metrics, new_carry)), grads = nnx.value_and_grad(
            objective,
            argnums=grad_argnums,
            has_aux=True,
        )(
            model,
            puzzle_embeddings,
            carry,
            batch,
            True,
            dropout_key,
        )
        if use_sparse_puzzle_embed:
            model_grads, puzzle_embedding_grads = grads
        else:
            model_grads = grads
        if use_sparse_puzzle_embed:
            optimizer.update(model, model_grads)
            touched_rows = update_sparse_puzzle_embeddings(
                model,
                puzzle_ids,
                puzzle_embedding_grads,
                learning_rate=puzzle_lr_schedule(optimizer_step),
                weight_decay=config.optimizer.puzzle_embed_weight_decay,
                coalesce_updates=config.optimizer.puzzle_embed_coalesce_updates,
            )
            metrics["puzzle_embed_learning_rate"] = puzzle_lr_schedule(optimizer_step)
            metrics["puzzle_embed_touched_rows"] = touched_rows
        else:
            optimizer.update(model, model_grads)
        return metrics, new_carry

    return nnx.jit(train_step)


def build_trm_dense_unroll_train_step_runner(
    *,
    halt_loss_weight: float = 0.0,
):
    def train_step(
        model: TinyRecursiveModel,
        optimizer: nnx.Optimizer,
        batch: dict[str, jax.Array],
        dropout_key: jax.Array,
    ) -> dict[str, jax.Array]:
        def objective(model, batch, train, dropout_key):
            return trm_dense_unroll_loss_and_metrics(
                model,
                batch,
                train,
                dropout_key,
                halt_loss_weight=halt_loss_weight,
            )

        (_, metrics), grads = nnx.value_and_grad(objective, has_aux=True)(
            model,
            batch,
            True,
            dropout_key,
        )
        optimizer.update(model, grads)
        return metrics

    return nnx.jit(train_step)


def build_trm_eval_step_runner(halt_loss_weight: float = 0.5, *, collect_diagnostics: bool = False):
    def eval_step(
        model: TinyRecursiveModel,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        _, metrics = trm_eval_loss_and_metrics(
            model,
            batch,
            halt_loss_weight,
            collect_diagnostics=collect_diagnostics,
        )
        return metrics

    return nnx.jit(eval_step)
