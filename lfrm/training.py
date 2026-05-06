from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from orbax.checkpoint import Checkpointer, PyTreeCheckpointHandler, args as ocp_args

from lfrm.config import ExperimentConfig
from lfrm.models import LatentFactorRecurrentModel, TinyRecursiveModel


GridReasoningModel = LatentFactorRecurrentModel | TinyRecursiveModel


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
    optimizer_steps = max(1, config.train.max_steps)
    warmup_steps = max(1, config.optimizer.warmup_steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.optimizer.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max(optimizer_steps, warmup_steps + 1),
        end_value=config.optimizer.learning_rate * config.optimizer.lr_min_ratio,
    )
    optimizer_schedule = lambda count: schedule(count + jnp.asarray(1, dtype=count.dtype))
    transforms = []
    if config.optimizer.grad_clip_norm > 0.0:
        transforms.append(optax.clip_by_global_norm(config.optimizer.grad_clip_norm))
    transforms.append(
        optax.adamw(
            learning_rate=optimizer_schedule,
            b1=config.optimizer.beta1,
            b2=config.optimizer.beta2,
            weight_decay=config.optimizer.weight_decay,
        )
    )
    optimizer = optax.chain(*transforms)
    if config.optimizer.flatten_optimizer:
        optimizer = optax.flatten(optimizer)
    return optimizer


def create_model(config: ExperimentConfig) -> GridReasoningModel:
    if config.model.model_type == "lfrm":
        return LatentFactorRecurrentModel(
            config.model,
            config.runtime,
            rngs=nnx.Rngs(config.train.seed),
        )
    if config.model.model_type == "trm":
        return TinyRecursiveModel(
            config.model,
            config.runtime,
            rngs=nnx.Rngs(config.train.seed),
        )
    raise ValueError("Only model_type='lfrm' or model_type='trm' is supported")


def create_optimizer(model: GridReasoningModel, config: ExperimentConfig) -> nnx.Optimizer:
    return nnx.Optimizer(model, build_optimizer(config), wrt=nnx.Param)


def create_ema_model(model: GridReasoningModel, config: ExperimentConfig) -> GridReasoningModel:
    """Create an eval-only shadow model initialized from the current params."""
    ema_model = create_model(config)
    nnx.update(ema_model, nnx.state(model))
    return ema_model


def ema_param_filter(config: ExperimentConfig):
    return nnx.Param


def build_ema_update_runner(decay: float, wrt=nnx.Param):
    def update_ema_model(ema_model: GridReasoningModel, model: GridReasoningModel) -> None:
        ema_params = nnx.state(ema_model, wrt)
        model_params = nnx.state(model, wrt)
        nnx.update(ema_model, nnx.state(model))
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
    step_loss_weighting: str = "uniform",
    q_loss_weight: float = 0.0,
    terminal_residual_weight: float = 0.0,
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

    use_final_forward = step_loss_weighting == "final" and hasattr(model, "forward_final_with_diagnostics")
    compute_terminal_residual = (not train) or terminal_residual_weight != 0.0
    model_forward_kwargs = {}
    if isinstance(model, (LatentFactorRecurrentModel, TinyRecursiveModel)):
        model_forward_kwargs = {
            "compute_terminal_residual": compute_terminal_residual,
        }
    if isinstance(model, TinyRecursiveModel):
        model_forward_kwargs["puzzle_identifiers"] = batch["puzzle_identifiers"]
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
    per_example_normalizer = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1.0)
    per_step_example_loss = jnp.sum(token_loss * step_loss_mask, axis=-1) / per_example_normalizer[None, :]

    step_weights = step_loss_weights(effective_step_logits.shape[0], step_loss_weighting)
    blank_ce_loss = jnp.sum(step_weights * per_step_loss)
    step_predictions = jnp.argmax(effective_step_logits, axis=-1)
    step_correct = (step_predictions == step_targets).astype(jnp.float32) * step_loss_mask
    per_step_example_accuracy = jnp.sum(step_correct, axis=-1) / per_example_normalizer[None, :]
    blanks_per_example = jnp.sum(loss_mask, axis=-1)
    step_correct_per_example = jnp.sum(step_correct, axis=-1)
    per_step_example_solved = jnp.where(
        blanks_per_example[None, :] > 0,
        step_correct_per_example == blanks_per_example[None, :],
        True,
    )
    quality_logits = diagnostics.get("quality_logits")
    q_loss = jnp.asarray(0.0, dtype=jnp.float32)
    q_selected_logits = effective_step_logits[-1]
    q_selected_step = jnp.full((inputs.shape[0],), effective_step_logits.shape[0] - 1, dtype=jnp.int32)
    if quality_logits is not None:
        if "quality_target_solved" in diagnostics:
            q_targets = jax.lax.stop_gradient(per_step_example_solved.astype(jnp.float32))
        else:
            q_targets = jax.lax.stop_gradient(per_step_example_accuracy)
        q_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(quality_logits, q_targets))
        q_selected_step = jnp.argmax(quality_logits, axis=0)
        gather_index = q_selected_step[None, :, None, None]
        q_selected_logits = jnp.take_along_axis(
            effective_step_logits,
            jnp.broadcast_to(gather_index, (1, inputs.shape[0], effective_step_logits.shape[2], effective_step_logits.shape[3])),
            axis=0,
        ).squeeze(0)
    terminal_residual = diagnostics.get(
        "terminal_belief_mse",
        diagnostics.get("terminal_belief_delta", jnp.asarray(0.0, dtype=jnp.float32)),
    )
    slot_consistency_loss = diagnostics.get("slot_consistency_loss", jnp.asarray(0.0, dtype=jnp.float32))
    slot_usage_loss = diagnostics.get("slot_usage_loss", jnp.asarray(0.0, dtype=jnp.float32))
    slot_diversity_loss = diagnostics.get("slot_diversity_loss", jnp.asarray(0.0, dtype=jnp.float32))
    loss = (
        blank_ce_loss
        + q_loss_weight * q_loss
        + terminal_residual_weight * terminal_residual
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

    correct_per_example = jnp.sum(correct, axis=-1)
    solved_examples = jnp.where(
        blanks_per_example > 0,
        correct_per_example == blanks_per_example,
        True,
    )
    solved_rate = jnp.mean(solved_examples.astype(jnp.float32))
    solved_count = jnp.sum(solved_examples.astype(jnp.float32))
    q_selected_token_loss = optax.softmax_cross_entropy_with_integer_labels(q_selected_logits, targets)
    q_selected_ce_loss = jnp.sum(q_selected_token_loss * loss_mask) / normalizer
    q_selected_predictions = jnp.argmax(q_selected_logits, axis=-1)
    q_selected_correct = (q_selected_predictions == targets).astype(jnp.float32) * loss_mask
    q_selected_accuracy = jnp.sum(q_selected_correct) / normalizer
    q_selected_correct_per_example = jnp.sum(q_selected_correct, axis=-1)
    q_selected_solved_examples = jnp.where(
        blanks_per_example > 0,
        q_selected_correct_per_example == blanks_per_example,
        True,
    )
    q_selected_solved_rate = jnp.mean(q_selected_solved_examples.astype(jnp.float32))
    oracle_step = jnp.argmin(per_step_example_loss, axis=0)

    metrics = {
        "loss": loss,
        "blank_ce_loss": blank_ce_loss,
        "q_loss": q_loss,
        "final_blank_ce_loss": per_step_loss[-1],
        "target_probability": target_probability,
        "blank_cell_accuracy": blank_cell_accuracy,
        "solved_rate": solved_rate,
        "q_selected_blank_ce_loss": q_selected_ce_loss,
        "q_selected_blank_cell_accuracy": q_selected_accuracy,
        "q_selected_solved_rate": q_selected_solved_rate,
        "q_selected_step": jnp.mean(q_selected_step.astype(jnp.float32) + 1.0),
        "oracle_step": jnp.mean(oracle_step.astype(jnp.float32) + 1.0),
        "per_step_loss": per_step_loss,
        "per_step_hidden_delta": diagnostics["hidden_delta_mean"],
    }
    if quality_logits is not None:
        metrics["per_step_quality_score"] = jax.nn.sigmoid(jnp.mean(quality_logits, axis=1))
    if "terminal_belief_delta" in diagnostics or terminal_residual_weight != 0.0:
        metrics["terminal_belief_delta"] = diagnostics.get("terminal_belief_delta", terminal_residual)
    if "terminal_belief_mse" in diagnostics or terminal_residual_weight != 0.0:
        metrics["terminal_belief_mse"] = terminal_residual
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
        "q_loss",
        "q_selected_blank_ce_loss",
        "q_selected_blank_cell_accuracy",
        "q_selected_solved_rate",
        "q_selected_step",
        "oracle_step",
        "slot_consistency_loss",
        "slot_usage_entropy",
        "slot_usage_loss",
        "slot_diversity_loss",
    ):
        if key in diagnostics:
            metrics[key] = diagnostics[key]
    return loss, metrics


def trm_act_loss_and_metrics(
    model: TinyRecursiveModel,
    carry: dict[str, jax.Array],
    batch: dict[str, jax.Array],
    train: bool,
    dropout_key: jax.Array | None,
    q_loss_weight: float = 0.5,
    puzzle_embeddings: jax.Array | None = None,
) -> tuple[jax.Array, tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
    new_carry, logits, diagnostics = model.forward_act_step(
        carry,
        batch,
        train=train,
        dropout_key=dropout_key,
        puzzle_embeddings=puzzle_embeddings,
    )
    targets = new_carry["current_labels"]
    given_mask = new_carry["current_given_mask"]
    loss_mask = (~given_mask).astype(jnp.float32)
    normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    per_example_normalizer = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1.0)

    token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    per_example_loss = jnp.sum(token_loss * loss_mask, axis=-1) / per_example_normalizer
    blank_ce_loss = jnp.mean(per_example_loss)

    predictions = jnp.argmax(logits, axis=-1)
    correct = (predictions == targets).astype(jnp.float32) * loss_mask
    blank_cell_accuracy = jnp.sum(correct) / normalizer
    correct_per_example = jnp.sum(correct, axis=-1)
    blanks_per_example = jnp.sum(loss_mask, axis=-1)
    solved_examples = jnp.where(
        blanks_per_example > 0,
        correct_per_example == blanks_per_example,
        True,
    )
    solved_targets = jax.lax.stop_gradient(solved_examples.astype(jnp.float32))
    solved_rate = jnp.mean(solved_targets)
    solved_count = jnp.sum(solved_targets)

    q_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(diagnostics["quality_logits"], solved_targets))
    loss = blank_ce_loss + q_loss_weight * q_loss

    probs = jax.nn.softmax(logits, axis=-1)
    target_probability = jnp.take_along_axis(probs, targets[..., None], axis=-1).squeeze(-1)
    target_probability = jnp.sum(target_probability * loss_mask) / normalizer
    metrics = {
        "loss": loss,
        "blank_ce_loss": blank_ce_loss,
        "final_blank_ce_loss": blank_ce_loss,
        "target_probability": target_probability,
        "blank_cell_accuracy": blank_cell_accuracy,
        "solved_rate": solved_rate,
        "solved_count": solved_count,
        "q_loss": q_loss,
        "act_step": diagnostics["act_step"],
        "halted_rate": diagnostics["halted_rate"],
        "reset_rate": diagnostics["reset_rate"],
    }
    return loss, (metrics, new_carry)


def trm_dense_unroll_loss_and_metrics(
    model: TinyRecursiveModel,
    batch: dict[str, jax.Array],
    train: bool,
    dropout_key: jax.Array | None,
    *,
    dense_loss_weight: float = 0.5,
    final_loss_weight: float = 0.5,
    sequence_loss_weight: float = 0.0,
    sequence_loss_temperature: float = 0.5,
    q_loss_weight: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    inputs = batch["inputs"]
    targets = batch["labels"]
    given_mask = batch["given_mask"]
    loss_mask = (~given_mask).astype(jnp.float32)
    normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    per_example_normalizer = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1.0)

    step_logits, diagnostics = model.forward_all_steps_with_diagnostics(
        inputs,
        puzzle_identifiers=batch["puzzle_identifiers"],
        train=train,
        dropout_key=dropout_key,
        compute_terminal_residual=False,
        include_layer_diagnostics=False,
    )
    step_targets = jnp.broadcast_to(targets[None, :, :], step_logits.shape[:-1])
    step_loss_mask = loss_mask[None, :, :]
    token_loss = optax.softmax_cross_entropy_with_integer_labels(step_logits, step_targets)
    per_step_loss = jnp.sum(token_loss * step_loss_mask, axis=(1, 2)) / normalizer
    num_steps = per_step_loss.shape[0]
    step_weights = jnp.full(
        (num_steps,),
        dense_loss_weight / jnp.asarray(num_steps, dtype=jnp.float32),
        dtype=jnp.float32,
    )
    step_weights = step_weights.at[-1].add(final_loss_weight)
    blank_ce_loss = jnp.sum(step_weights * per_step_loss)
    mean_blank_ce_loss = jnp.mean(per_step_loss)
    final_blank_ce_loss = per_step_loss[-1]
    masked_token_loss = jnp.where(step_loss_mask.astype(bool), token_loss / sequence_loss_temperature, -1e9)
    blanks_per_example = jnp.sum(loss_mask, axis=-1)
    per_step_sequence_loss = sequence_loss_temperature * (
        jax.nn.logsumexp(masked_token_loss, axis=-1)
        - jnp.log(jnp.maximum(blanks_per_example[None, :], 1.0))
    )
    per_step_sequence_loss = jnp.where(blanks_per_example[None, :] > 0, per_step_sequence_loss, 0.0)
    sequence_loss = jnp.sum(step_weights * jnp.mean(per_step_sequence_loss, axis=-1))

    step_predictions = jnp.argmax(step_logits, axis=-1)
    step_correct = (step_predictions == step_targets).astype(jnp.float32) * step_loss_mask
    per_step_accuracy = jnp.sum(step_correct, axis=(1, 2)) / normalizer
    step_correct_per_example = jnp.sum(step_correct, axis=-1)
    per_step_example_solved = jnp.where(
        blanks_per_example[None, :] > 0,
        step_correct_per_example == blanks_per_example[None, :],
        True,
    )
    final_logits = step_logits[-1]
    predictions = jnp.argmax(final_logits, axis=-1)
    final_probs = jax.nn.softmax(final_logits, axis=-1)
    target_probability = jnp.take_along_axis(final_probs, targets[..., None], axis=-1).squeeze(-1)
    target_probability = jnp.sum(target_probability * loss_mask) / normalizer
    correct = (predictions == targets).astype(jnp.float32) * loss_mask
    blank_cell_accuracy = jnp.sum(correct) / normalizer
    correct_per_example = jnp.sum(correct, axis=-1)
    solved_examples = jnp.where(
        blanks_per_example > 0,
        correct_per_example == blanks_per_example,
        True,
    )
    solved_rate = jnp.mean(solved_examples.astype(jnp.float32))
    solved_count = jnp.sum(solved_examples.astype(jnp.float32))

    quality_logits = diagnostics.get("quality_logits")
    q_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if quality_logits is not None and q_loss_weight != 0.0:
        q_targets = jax.lax.stop_gradient(per_step_example_solved.astype(jnp.float32))
        q_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(quality_logits, q_targets))

    loss = blank_ce_loss + sequence_loss_weight * sequence_loss + q_loss_weight * q_loss
    oracle_step = jnp.argmin(
        jnp.sum(token_loss * step_loss_mask, axis=-1) / per_example_normalizer[None, :],
        axis=0,
    )
    metrics = {
        "loss": loss,
        "blank_ce_loss": blank_ce_loss,
        "mean_blank_ce_loss": mean_blank_ce_loss,
        "final_blank_ce_loss": final_blank_ce_loss,
        "sequence_loss": sequence_loss,
        "target_probability": target_probability,
        "blank_cell_accuracy": blank_cell_accuracy,
        "solved_rate": solved_rate,
        "solved_count": solved_count,
        "q_loss": q_loss,
        "oracle_step": jnp.mean(oracle_step.astype(jnp.float32) + 1.0),
        "per_step_loss": per_step_loss,
        "per_step_hidden_delta": diagnostics["hidden_delta_mean"],
        "unroll_steps": jnp.asarray(step_logits.shape[0], dtype=jnp.float32),
    }
    if quality_logits is not None:
        metrics["per_step_quality_score"] = jax.nn.sigmoid(jnp.mean(quality_logits, axis=1))
    return loss, metrics


def trm_eval_loss_and_metrics(
    model: TinyRecursiveModel,
    batch: dict[str, jax.Array],
    q_loss_weight: float = 0.5,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    inputs = batch["inputs"]
    targets = batch["labels"]
    given_mask = batch["given_mask"]
    loss_mask = (~given_mask).astype(jnp.float32)
    normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    per_example_normalizer = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1.0)

    step_logits, diagnostics = model.forward_all_steps_with_diagnostics(
        inputs,
        puzzle_identifiers=batch["puzzle_identifiers"],
        train=False,
        dropout_key=None,
        compute_terminal_residual=False,
        include_layer_diagnostics=True,
    )
    step_targets = jnp.broadcast_to(targets[None, :, :], step_logits.shape[:-1])
    step_loss_mask = loss_mask[None, :, :]
    token_loss = optax.softmax_cross_entropy_with_integer_labels(step_logits, step_targets)
    per_step_loss = jnp.sum(token_loss * step_loss_mask, axis=(1, 2)) / normalizer
    per_step_example_loss = jnp.sum(token_loss * step_loss_mask, axis=-1) / per_example_normalizer[None, :]

    step_predictions = jnp.argmax(step_logits, axis=-1)
    step_correct = (step_predictions == step_targets).astype(jnp.float32) * step_loss_mask
    per_step_accuracy = jnp.sum(step_correct, axis=(1, 2)) / normalizer
    step_correct_per_example = jnp.sum(step_correct, axis=-1)
    blanks_per_example = jnp.sum(loss_mask, axis=-1)
    per_step_example_solved = jnp.where(
        blanks_per_example[None, :] > 0,
        step_correct_per_example == blanks_per_example[None, :],
        True,
    )
    l_logits = diagnostics.get("l_logits")
    l_per_step_loss = jnp.zeros_like(per_step_loss)
    l_per_step_accuracy = jnp.zeros_like(per_step_loss)
    if l_logits is not None:
        l_token_loss = optax.softmax_cross_entropy_with_integer_labels(l_logits, step_targets)
        l_per_step_loss = jnp.sum(l_token_loss * step_loss_mask, axis=(1, 2)) / normalizer
        l_predictions = jnp.argmax(l_logits, axis=-1)
        l_correct = (l_predictions == step_targets).astype(jnp.float32) * step_loss_mask
        l_per_step_accuracy = jnp.sum(l_correct, axis=(1, 2)) / normalizer
    quality_logits = diagnostics["quality_logits"]
    q_continue_logits = diagnostics.get("q_continue_logits")
    if model.trm.no_act_continue or q_continue_logits is None:
        step_halted = quality_logits > 0.0
    else:
        step_halted = quality_logits > q_continue_logits
    step_halted = step_halted.at[-1, :].set(True)
    selected_step = jnp.argmax(step_halted.astype(jnp.int32), axis=0)
    gather_index = selected_step[None, :, None, None]
    selected_logits = jnp.take_along_axis(
        step_logits,
        jnp.broadcast_to(gather_index, (1, inputs.shape[0], step_logits.shape[2], step_logits.shape[3])),
        axis=0,
    ).squeeze(0)
    selected_token_loss = optax.softmax_cross_entropy_with_integer_labels(selected_logits, targets)
    blank_ce_loss = jnp.sum(selected_token_loss * loss_mask) / normalizer
    selected_solved_targets = jnp.take_along_axis(
        per_step_example_solved.astype(jnp.float32),
        selected_step[None, :],
        axis=0,
    ).squeeze(0)
    selected_solved_count = jnp.sum(selected_solved_targets)
    q_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(quality_logits, jax.lax.stop_gradient(per_step_example_solved.astype(jnp.float32))))
    loss = blank_ce_loss + q_loss_weight * q_loss

    predictions = jnp.argmax(selected_logits, axis=-1)
    selected_probs = jax.nn.softmax(selected_logits, axis=-1)
    target_probability = jnp.take_along_axis(selected_probs, targets[..., None], axis=-1).squeeze(-1)
    target_probability = jnp.sum(target_probability * loss_mask) / normalizer
    correct = (predictions == targets).astype(jnp.float32) * loss_mask
    blank_cell_accuracy = jnp.sum(correct) / normalizer
    solved_rate = jnp.mean(selected_solved_targets)
    final_logits = step_logits[-1]
    final_token_loss = optax.softmax_cross_entropy_with_integer_labels(final_logits, targets)
    final_blank_ce_loss = jnp.sum(final_token_loss * loss_mask) / normalizer
    final_predictions = jnp.argmax(final_logits, axis=-1)
    final_correct = (final_predictions == targets).astype(jnp.float32) * loss_mask
    final_blank_cell_accuracy = jnp.sum(final_correct) / normalizer
    final_correct_per_example = jnp.sum(final_correct, axis=-1)
    final_solved_examples = jnp.where(
        blanks_per_example > 0,
        final_correct_per_example == blanks_per_example,
        True,
    )
    final_solved_count = jnp.sum(final_solved_examples.astype(jnp.float32))
    oracle_step = jnp.argmin(per_step_example_loss, axis=0)
    metrics = {
        "loss": loss,
        "blank_ce_loss": blank_ce_loss,
        "q_loss": q_loss,
        "final_blank_ce_loss": final_blank_ce_loss,
        "final_blank_cell_accuracy": final_blank_cell_accuracy,
        "final_solved_rate": jnp.mean(final_solved_examples.astype(jnp.float32)),
        "target_probability": target_probability,
        "blank_cell_accuracy": blank_cell_accuracy,
        "solved_rate": solved_rate,
        "solved_count": selected_solved_count,
        "q_selected_blank_ce_loss": blank_ce_loss,
        "q_selected_blank_cell_accuracy": blank_cell_accuracy,
        "q_selected_solved_rate": solved_rate,
        "q_selected_solved_count": selected_solved_count,
        "q_selected_step": jnp.mean(selected_step.astype(jnp.float32) + 1.0),
        "oracle_step": jnp.mean(oracle_step.astype(jnp.float32) + 1.0),
        "final_solved_count": final_solved_count,
        "per_step_loss": per_step_loss,
        "per_step_h_loss": per_step_loss,
        "per_step_l_loss": l_per_step_loss,
        "per_step_h_accuracy": per_step_accuracy,
        "per_step_l_accuracy": l_per_step_accuracy,
        "per_step_hidden_delta": diagnostics["hidden_delta_mean"],
        "per_step_h_hidden_delta": diagnostics.get("h_hidden_delta_mean", diagnostics["hidden_delta_mean"]),
        "per_step_l_hidden_delta": diagnostics.get("l_hidden_delta_mean", jnp.zeros_like(diagnostics["hidden_delta_mean"])),
        "per_step_quality_score": jax.nn.sigmoid(jnp.mean(quality_logits, axis=1)),
        "unroll_steps": jnp.mean(selected_step.astype(jnp.float32) + 1.0),
    }
    return loss, metrics


def build_train_step_runner(
    step_loss_weighting: str = "uniform",
    q_loss_weight: float = 0.0,
    terminal_residual_weight: float = 0.0,
    slot_consistency_weight: float = 0.0,
    slot_usage_weight: float = 0.0,
    slot_diversity_weight: float = 0.0,
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
                step_loss_weighting,
                q_loss_weight,
                terminal_residual_weight,
                slot_consistency_weight,
                slot_usage_weight,
                slot_diversity_weight,
            )

        grad_fn = nnx.value_and_grad(weighted_loss_and_metrics, has_aux=True)
        (_, metrics), grads = grad_fn(model, batch, True, dropout_key)
        optimizer.update(model, grads)
        return metrics

    return nnx.jit(train_step_with_weight)


def build_trm_act_train_step_runner(q_loss_weight: float = 0.5):
    def train_step(
        model: TinyRecursiveModel,
        optimizer: nnx.Optimizer,
        carry: dict[str, jax.Array],
        batch: dict[str, jax.Array],
        dropout_key: jax.Array,
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        def objective(model, carry, batch, train, dropout_key):
            return trm_act_loss_and_metrics(
                model,
                carry,
                batch,
                train,
                dropout_key,
                q_loss_weight,
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

    return nnx.jit(train_step)


def build_trm_dense_unroll_train_step_runner(
    *,
    dense_loss_weight: float = 0.5,
    final_loss_weight: float = 0.5,
    sequence_loss_weight: float = 0.0,
    sequence_loss_temperature: float = 0.5,
    q_loss_weight: float = 0.0,
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
                dense_loss_weight=dense_loss_weight,
                final_loss_weight=final_loss_weight,
                sequence_loss_weight=sequence_loss_weight,
                sequence_loss_temperature=sequence_loss_temperature,
                q_loss_weight=q_loss_weight,
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


def build_eval_step_runner(
    step_loss_weighting: str = "uniform",
    q_loss_weight: float = 0.0,
    terminal_residual_weight: float = 0.0,
    slot_consistency_weight: float = 0.0,
    slot_usage_weight: float = 0.0,
    slot_diversity_weight: float = 0.0,
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
            step_loss_weighting,
            q_loss_weight,
            terminal_residual_weight,
            slot_consistency_weight,
            slot_usage_weight,
            slot_diversity_weight,
        )
        return metrics

    return nnx.jit(eval_step_with_weight)


def build_trm_eval_step_runner(q_loss_weight: float = 0.5):
    def eval_step(
        model: TinyRecursiveModel,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        _, metrics = trm_eval_loss_and_metrics(
            model,
            batch,
            q_loss_weight,
        )
        return metrics

    return nnx.jit(eval_step)


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
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": step,
        "uses_ema_model": ema_model is not None,
    }
    if ema_model is not None:
        payload["ema_model"] = nnx.state(ema_model)
    checkpointer.save(target_dir, payload, force=True)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: GridReasoningModel,
    optimizer: nnx.Optimizer,
    *,
    ema_model: GridReasoningModel | None = None,
) -> int:
    checkpointer = Checkpointer(PyTreeCheckpointHandler())
    restore_target = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": 0,
        "uses_ema_model": False,
    }
    if ema_model is not None:
        restore_target["ema_model"] = nnx.state(ema_model)
    payload = checkpointer.restore(
        Path(checkpoint_path).resolve(),
        args=ocp_args.PyTreeRestore(item=restore_target, partial_restore=True),
    )
    nnx.update(model, payload["model"])
    nnx.update(optimizer, payload["optimizer"])
    if ema_model is not None:
        if "ema_model" in payload:
            nnx.update(ema_model, payload["ema_model"])
        else:
            nnx.update(ema_model, payload["model"])
    return int(payload["step"])
