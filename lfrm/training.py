from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from orbax.checkpoint import Checkpointer, PyTreeCheckpointHandler

from lfrm.config import ExperimentConfig
from lfrm.models import LatentFactorRecurrentModel


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
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.optimizer.grad_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=config.optimizer.weight_decay,
        ),
    )
    if config.optimizer.flatten_optimizer:
        return optax.flatten(optimizer)
    return optimizer


def create_model(config: ExperimentConfig) -> LatentFactorRecurrentModel:
    if config.model.model_type != "lfrm":
        raise ValueError("Only model_type='lfrm' is supported")
    return LatentFactorRecurrentModel(
        config.model,
        config.runtime,
        rngs=nnx.Rngs(config.train.seed),
    )


def create_optimizer(model: LatentFactorRecurrentModel, config: ExperimentConfig) -> nnx.Optimizer:
    return nnx.Optimizer(model, build_optimizer(config), wrt=nnx.Param)


def create_ema_model(model: LatentFactorRecurrentModel, config: ExperimentConfig) -> LatentFactorRecurrentModel:
    """Create an eval-only shadow model initialized from the current params."""
    ema_model = create_model(config)
    nnx.update(ema_model, nnx.state(model, nnx.Param))
    return ema_model


def build_ema_update_runner(decay: float):
    def update_ema_model(ema_model: LatentFactorRecurrentModel, model: LatentFactorRecurrentModel) -> None:
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
    model: LatentFactorRecurrentModel,
    batch: dict[str, jax.Array],
    train: bool,
    dropout_key: jax.Array | None,
    step_loss_weighting: str = "uniform",
    terminal_residual_weight: float = 0.0,
    energy_loss_weight: float = 0.0,
    energy_margin: float = 1.0,
    energy_corruptions: int = 1,
    slot_consistency_weight: float = 0.0,
    slot_usage_weight: float = 0.0,
    slot_diversity_weight: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    inputs = batch["inputs"]
    targets = batch["labels"]
    given_mask = batch["given_mask"]
    loss_mask = (~given_mask).astype(jnp.float32)
    normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)

    model_key = dropout_key
    energy_key = jax.random.key(0)
    if dropout_key is not None:
        model_key, energy_key = jax.random.split(dropout_key)

    use_final_forward = step_loss_weighting == "final" and hasattr(model, "forward_final_with_diagnostics")
    compute_energy_selection = energy_loss_weight != 0.0
    if isinstance(model, LatentFactorRecurrentModel) and not train:
        compute_energy_selection = compute_energy_selection or model.lfrm.num_branches > 1
    compute_terminal_residual = (not train) or terminal_residual_weight != 0.0
    model_forward_kwargs = {}
    if isinstance(model, LatentFactorRecurrentModel):
        model_forward_kwargs = {
            "compute_energy_selection": compute_energy_selection,
            "compute_terminal_residual": compute_terminal_residual,
        }
    if use_final_forward:
        step_logits, diagnostics = model.forward_final_with_diagnostics(
            inputs,
            train=train,
            dropout_key=model_key,
            **model_forward_kwargs,
        )
    else:
        step_logits, diagnostics = model.forward_all_steps_with_diagnostics(
            inputs,
            train=train,
            dropout_key=model_key,
            **model_forward_kwargs,
        )
    effective_step_logits = step_logits
    step_targets = jnp.broadcast_to(targets[None, :, :], step_logits.shape[:-1])
    step_loss_mask = loss_mask[None, :, :]
    token_loss = optax.softmax_cross_entropy_with_integer_labels(effective_step_logits, step_targets)
    per_step_loss = jnp.sum(token_loss * step_loss_mask, axis=(1, 2)) / normalizer

    step_weights = step_loss_weights(effective_step_logits.shape[0], step_loss_weighting)
    blank_ce_loss = jnp.sum(step_weights * per_step_loss)
    branch_min_ce = None
    branch_mean_ce = None
    if "branch_logits" in diagnostics:
        branch_logits = diagnostics.get("branch_digit_logits", diagnostics["branch_logits"])
        branch_targets_raw = targets - 2 if "branch_digit_logits" in diagnostics else targets
        branch_targets = jnp.broadcast_to(branch_targets_raw[:, None, :], branch_logits.shape[:-1])
        branch_loss_mask = loss_mask[:, None, :]
        branch_token_loss = optax.softmax_cross_entropy_with_integer_labels(branch_logits, branch_targets)
        per_example_normalizer = jnp.maximum(jnp.sum(loss_mask, axis=-1, keepdims=True), 1.0)
        branch_ce = jnp.sum(branch_token_loss * branch_loss_mask, axis=-1) / per_example_normalizer
        blank_ce_loss = jnp.mean(branch_ce)
        branch_min_ce = jnp.mean(jnp.min(branch_ce, axis=-1))
        branch_mean_ce = jnp.mean(branch_ce)
    terminal_residual = diagnostics.get(
        "terminal_belief_mse",
        diagnostics.get("terminal_belief_delta", jnp.asarray(0.0, dtype=jnp.float32)),
    )
    slot_consistency_loss = diagnostics.get("slot_consistency_loss", jnp.asarray(0.0, dtype=jnp.float32))
    slot_usage_loss = diagnostics.get("slot_usage_loss", jnp.asarray(0.0, dtype=jnp.float32))
    slot_diversity_loss = diagnostics.get("slot_diversity_loss", jnp.asarray(0.0, dtype=jnp.float32))
    energy_metrics = {}
    if hasattr(model, "energy_training_metrics") and energy_loss_weight != 0.0:
        energy_metrics = model.energy_training_metrics(
            inputs,
            targets,
            given_mask,
            energy_key,
            diagnostics,
            margin=energy_margin,
            corruptions=energy_corruptions,
        )
    energy_margin_loss = energy_metrics.get("energy_margin_loss", jnp.asarray(0.0, dtype=jnp.float32))
    loss = (
        blank_ce_loss
        + terminal_residual_weight * terminal_residual
        + energy_loss_weight * energy_margin_loss
        + slot_consistency_weight * slot_consistency_loss
        + slot_usage_weight * slot_usage_loss
        + slot_diversity_weight * slot_diversity_loss
    )

    final_logits = effective_step_logits[-1]
    predictions = jnp.argmax(final_logits, axis=-1)
    final_probs = jax.nn.softmax(final_logits, axis=-1)
    target_probability = jnp.take_along_axis(final_probs, targets[..., None], axis=-1).squeeze(-1)
    target_probability = jnp.sum(target_probability * loss_mask) / normalizer
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
        "target_probability": target_probability,
        "blank_cell_accuracy": blank_cell_accuracy,
        "solved_rate": solved_rate,
        "per_step_loss": per_step_loss,
        "per_step_hidden_delta": diagnostics["hidden_delta_mean"],
    }
    if "terminal_belief_delta" in diagnostics or terminal_residual_weight != 0.0:
        metrics["terminal_belief_delta"] = diagnostics.get("terminal_belief_delta", terminal_residual)
    if "terminal_belief_mse" in diagnostics or terminal_residual_weight != 0.0:
        metrics["terminal_belief_mse"] = terminal_residual
    if energy_loss_weight != 0.0:
        metrics["energy_margin_loss"] = energy_margin_loss
    if branch_min_ce is not None:
        metrics["branch_min_ce"] = branch_min_ce
    if branch_mean_ce is not None:
        metrics["branch_mean_ce"] = branch_mean_ce
    if "rho_mean" in diagnostics:
        metrics["per_step_rho"] = diagnostics["rho_mean"]
    if "alpha_mean" in diagnostics:
        metrics["per_step_alpha"] = diagnostics["alpha_mean"]
    for key in (
        "unroll_steps",
        "terminal_belief_delta",
        "terminal_belief_mse",
        "belief_entropy",
        "belief_confidence",
        "selected_branch_energy",
        "slot_consistency_loss",
        "slot_usage_entropy",
        "slot_usage_loss",
        "slot_diversity_loss",
        "branch_diversity",
        "final_branch_diversity",
    ):
        if key in diagnostics:
            metrics[key] = diagnostics[key]
    for key, value in energy_metrics.items():
        metrics[key] = value
    return loss, metrics


def build_train_step_runner(
    step_loss_weighting: str = "uniform",
    terminal_residual_weight: float = 0.0,
    energy_loss_weight: float = 0.0,
    energy_margin: float = 1.0,
    energy_corruptions: int = 1,
    slot_consistency_weight: float = 0.0,
    slot_usage_weight: float = 0.0,
    slot_diversity_weight: float = 0.0,
):
    def train_step_with_weight(
        model: LatentFactorRecurrentModel,
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
                step_loss_weighting,
                terminal_residual_weight,
                energy_loss_weight,
                energy_margin,
                energy_corruptions,
                slot_consistency_weight,
                slot_usage_weight,
                slot_diversity_weight,
            )

        grad_fn = nnx.value_and_grad(weighted_loss_and_metrics, has_aux=True)
        (_, metrics), grads = grad_fn(model, batch, True, dropout_key)
        optimizer.update(model, grads)
        return metrics

    return nnx.jit(train_step_with_weight)


def build_eval_step_runner(
    step_loss_weighting: str = "uniform",
    terminal_residual_weight: float = 0.0,
    energy_loss_weight: float = 0.0,
    energy_margin: float = 1.0,
    energy_corruptions: int = 1,
    slot_consistency_weight: float = 0.0,
    slot_usage_weight: float = 0.0,
    slot_diversity_weight: float = 0.0,
):
    def eval_step_with_weight(
        model: LatentFactorRecurrentModel,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        _, metrics = loss_and_metrics(
            model,
            batch,
            False,
            None,
            step_loss_weighting,
            terminal_residual_weight,
            energy_loss_weight,
            energy_margin,
            energy_corruptions,
            slot_consistency_weight,
            slot_usage_weight,
            slot_diversity_weight,
        )
        return metrics

    return nnx.jit(eval_step_with_weight)


def save_checkpoint(
    checkpoint_dir: str,
    model: LatentFactorRecurrentModel,
    optimizer: nnx.Optimizer,
    step: int,
    *,
    ema_model: LatentFactorRecurrentModel | None = None,
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
