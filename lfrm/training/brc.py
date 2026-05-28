from __future__ import annotations

import jax
from flax import nnx

from lfrm.models import BRCModel
from lfrm.training.factory import GridReasoningModel
from lfrm.training.steps import brc_carry_loss_and_metrics, loss_and_metrics


def build_train_step_runner(
    halt_loss_weight: float = 0.0,
    terminal_residual_weight: float = 0.0,
):
    def train_step_with_weight(
        model: GridReasoningModel,
        optimizer: nnx.Optimizer,
        batch: dict[str, jax.Array],
        dropout_key: jax.Array,
    ) -> dict[str, jax.Array]:
        def weighted_loss_and_metrics(model, batch, train, dropout_key):
            return loss_and_metrics(
                model,
                batch,
                train,
                dropout_key,
                halt_loss_weight,
                terminal_residual_weight,
            )

        grad_fn = nnx.value_and_grad(weighted_loss_and_metrics, has_aux=True)
        (_, metrics), grads = grad_fn(model, batch, True, dropout_key)
        optimizer.update(model, grads)
        return metrics

    return nnx.jit(train_step_with_weight)


def build_brc_step_carry_train_step_runner():
    def train_step(
        model: BRCModel,
        optimizer: nnx.Optimizer,
        carry: dict[str, jax.Array],
        batch: dict[str, jax.Array],
        dropout_key: jax.Array,
        optimizer_step: jax.Array,
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        dropout_key = jax.random.fold_in(dropout_key, optimizer_step)

        def objective(model, carry, batch, train, dropout_key):
            return brc_carry_loss_and_metrics(
                model,
                carry,
                batch,
                train,
                dropout_key,
            )

        (_, (metrics, new_carry)), grads = nnx.value_and_grad(objective, has_aux=True)(
            model,
            carry,
            batch,
            True,
            dropout_key,
        )
        optimizer.update(model, grads)
        return metrics, new_carry

    # Step-carry is a truncated execution path for BRC. It carries z/H between
    # optimizer updates and lets deterministic energy-stability early stop reset
    # samples without training a learned halt head.
    return nnx.jit(train_step)


def build_eval_step_runner(
    halt_loss_weight: float = 0.0,
    terminal_residual_weight: float = 0.0,
):
    def eval_step_with_weight(
        model: GridReasoningModel,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        _, metrics = loss_and_metrics(
            model,
            batch,
            False,
            None,
            halt_loss_weight,
            terminal_residual_weight,
        )
        return metrics

    # Eval batches are raw input/label pytrees; donation is not useful enough to
    # justify the unusable-buffer warnings on integer leaves.
    return nnx.jit(eval_step_with_weight)
