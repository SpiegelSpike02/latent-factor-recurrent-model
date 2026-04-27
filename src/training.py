from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from orbax.checkpoint import Checkpointer, PyTreeCheckpointHandler

from config import ExperimentConfig
from models import GridReasoningModel
from tasks.sudoku import apply_given_logits_by_step, soft_sudoku_validity_loss


def step_loss_weights(num_steps: int, weighting: str) -> jax.Array:
    if weighting == "uniform":
        return jnp.full((num_steps,), 1.0 / num_steps, dtype=jnp.float32)
    if weighting == "linear":
        weights = jnp.arange(1, num_steps + 1, dtype=jnp.float32)
        return weights / jnp.sum(weights)
    if weighting == "final":
        return jax.nn.one_hot(num_steps - 1, num_steps, dtype=jnp.float32)
    raise ValueError(f"Unsupported step_loss_weighting: {weighting}")


def build_optimizer(config: ExperimentConfig) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.optimizer.learning_rate,
        warmup_steps=config.optimizer.warmup_steps,
        decay_steps=max(config.train.max_steps, config.optimizer.warmup_steps + 1),
        end_value=config.optimizer.learning_rate * 0.1,
    )
    return optax.chain(
        optax.clip_by_global_norm(config.optimizer.grad_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=config.optimizer.weight_decay,
        ),
    )


def create_model(config: ExperimentConfig) -> GridReasoningModel:
    return GridReasoningModel(
        config.model,
        config.runtime,
        rngs=nnx.Rngs(config.train.seed),
    )


def create_optimizer(model: GridReasoningModel, config: ExperimentConfig) -> nnx.Optimizer:
    return nnx.Optimizer(model, build_optimizer(config), wrt=nnx.Param)


def create_ema_model(model: GridReasoningModel, config: ExperimentConfig) -> GridReasoningModel:
    """Create an eval-only shadow model initialized from the current params."""
    ema_model = create_model(config)
    nnx.update(ema_model, nnx.state(model, nnx.Param))
    return ema_model


def build_ema_update_runner(decay: float):
    def update_ema_model(ema_model: GridReasoningModel, model: GridReasoningModel) -> None:
        ema_params = nnx.state(ema_model, nnx.Param)
        model_params = nnx.state(model, nnx.Param)
        updated_params = jax.tree.map(
            lambda ema, current: decay * ema + (1.0 - decay) * current,
            ema_params,
            model_params,
        )
        nnx.update(ema_model, updated_params)

    return nnx.jit(update_ema_model)


def loss_and_metrics(
    model: GridReasoningModel,
    batch: dict[str, jax.Array],
    train: bool,
    dropout_key: jax.Array | None,
    validity_loss_weight: float = 0.0,
    step_loss_weighting: str = "uniform",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    inputs = batch["inputs"]
    targets = batch["labels"]
    given_mask = batch["given_mask"]
    loss_mask = (~given_mask).astype(jnp.float32)
    normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)

    step_logits, diagnostics = model.forward_all_steps_with_diagnostics(
        inputs,
        train=train,
        dropout_key=dropout_key,
    )
    if model.config.fix_clue_outputs:
        effective_step_logits = apply_given_logits_by_step(step_logits, inputs, given_mask)
    else:
        effective_step_logits = step_logits
    step_targets = jnp.broadcast_to(targets[None, :, :], step_logits.shape[:-1])
    step_loss_mask = loss_mask[None, :, :]
    token_loss = optax.softmax_cross_entropy_with_integer_labels(effective_step_logits, step_targets)
    per_step_loss = jnp.sum(token_loss * step_loss_mask, axis=(1, 2)) / normalizer

    step_weights = step_loss_weights(model.config.num_steps, step_loss_weighting)
    blank_ce_loss = jnp.sum(step_weights * per_step_loss)
    per_step_validity_loss = soft_sudoku_validity_loss(
        effective_step_logits,
        model.row_unit_matrix,
        model.col_unit_matrix,
        model.box_unit_matrix,
    )
    validity_loss = jnp.sum(step_weights * per_step_validity_loss)
    loss = blank_ce_loss + validity_loss_weight * validity_loss

    final_logits = effective_step_logits[-1]
    predictions = jnp.argmax(final_logits, axis=-1)
    correct = (predictions == targets).astype(jnp.float32) * loss_mask
    blank_cell_accuracy = jnp.sum(correct) / normalizer

    blanks_per_example = jnp.sum(loss_mask, axis=-1)
    correct_per_example = jnp.sum(correct, axis=-1)
    solved_examples = jnp.where(
        blanks_per_example > 0,
        correct_per_example == blanks_per_example,
        True,
    )
    solved_rate = jnp.mean(solved_examples.astype(jnp.float32))

    metrics = {
        "loss": loss,
        "blank_ce_loss": blank_ce_loss,
        "final_blank_ce_loss": per_step_loss[-1],
        "validity_loss": validity_loss,
        "blank_cell_accuracy": blank_cell_accuracy,
        "solved_rate": solved_rate,
        "per_step_loss": per_step_loss,
        "per_step_validity_loss": per_step_validity_loss,
        "per_step_hidden_delta": diagnostics["hidden_delta_mean"],
    }
    if "rho_mean" in diagnostics:
        metrics["per_step_rho"] = diagnostics["rho_mean"]
    if "alpha_mean" in diagnostics:
        metrics["per_step_alpha"] = diagnostics["alpha_mean"]
    return loss, metrics


def build_train_step_runner(validity_loss_weight: float, step_loss_weighting: str = "uniform"):
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
                validity_loss_weight,
                step_loss_weighting,
            )

        grad_fn = nnx.value_and_grad(weighted_loss_and_metrics, has_aux=True)
        (_, metrics), grads = grad_fn(model, batch, True, dropout_key)
        optimizer.update(model, grads)
        return metrics

    return nnx.jit(train_step_with_weight)


def build_eval_step_runner(validity_loss_weight: float, step_loss_weighting: str = "uniform"):
    def eval_step_with_weight(
        model: GridReasoningModel,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        _, metrics = loss_and_metrics(model, batch, False, None, validity_loss_weight, step_loss_weighting)
        return metrics

    return nnx.jit(eval_step_with_weight)


def save_checkpoint(
    checkpoint_dir: str,
    model: GridReasoningModel,
    optimizer: nnx.Optimizer,
    step: int,
    *,
    ema_model: GridReasoningModel | None = None,
) -> None:
    checkpointer = Checkpointer(PyTreeCheckpointHandler())
    target_dir = Path(checkpoint_dir).resolve() / f"step_{step}"
    payload = {
        "model": nnx.state(ema_model if ema_model is not None else model),
        "optimizer": nnx.state(optimizer),
        "step": step,
        "uses_ema_model": ema_model is not None,
    }
    checkpointer.save(target_dir, payload, force=True)
