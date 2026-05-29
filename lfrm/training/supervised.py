from __future__ import annotations

import jax
from flax import nnx

from lfrm.training.factory import GridReasoningModel
from lfrm.training.objectives import loss_and_metrics


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

    return nnx.jit(eval_step_with_weight)
