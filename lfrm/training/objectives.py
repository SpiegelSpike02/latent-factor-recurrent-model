from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from lfrm.models import BeliefDynamicsReasoner, TinyRecursiveModel, UnifiedReasoningModel
from lfrm.training.factory import GridReasoningModel
from lfrm.training.losses import (
    output_probabilities,
    supervised_loss_mask,
    target_probability as token_target_probability,
    token_cross_entropy,
)
from lfrm.training.metrics import maybe_path_metrics


def _maybe_path_metrics(
    model: GridReasoningModel,
    predictions: jax.Array,
    targets: jax.Array,
    loss_mask: jax.Array,
) -> dict[str, jax.Array]:
    return maybe_path_metrics(
        task_type=getattr(getattr(model, "config", None), "task_type", "sudoku"),
        predictions=predictions,
        targets=targets,
        loss_mask=loss_mask,
    )


def _example_mask(batch: dict[str, jax.Array], targets: jax.Array) -> jax.Array:
    if "example_mask" in batch:
        return batch["example_mask"].astype(jnp.float32)
    return jnp.ones_like(targets[:, 0], dtype=jnp.float32)


def _apply_example_mask(loss_mask: jax.Array, example_mask: jax.Array) -> jax.Array:
    return loss_mask.astype(jnp.float32) * example_mask[:, None]


def _masked_example_mean(values: jax.Array, example_mask: jax.Array) -> jax.Array:
    return jnp.sum(values.astype(jnp.float32) * example_mask) / jnp.maximum(jnp.sum(example_mask), 1.0)


def _permute_sudoku_digits(
    inputs: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    *,
    train: bool,
) -> tuple[jax.Array, jax.Array]:
    if not train:
        return inputs, targets
    perms = jax.vmap(jax.random.permutation, in_axes=(0, None))(
        jax.random.split(key, inputs.shape[0]),
        jnp.arange(9, dtype=jnp.int32),
    )

    input_digit_ids = jnp.clip(inputs - 1, 0, 8)
    permuted_inputs = jnp.take_along_axis(perms, input_digit_ids, axis=1) + 1
    inputs = jnp.where(inputs > 0, permuted_inputs, inputs).astype(jnp.int32)
    target_digit_ids = jnp.clip(targets, 0, 8)
    targets = jnp.take_along_axis(perms, target_digit_ids, axis=1).astype(jnp.int32)
    return inputs, targets


def _sudoku_board_metrics(
    model: GridReasoningModel,
    predictions: jax.Array,
    inputs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    if model.config.task_type == "sudoku":
        context_mask = inputs > 0
        context_targets = jnp.clip(inputs - 1, 0, 8)
        digit_indices = jnp.clip(predictions, 0, 8)
    else:
        context_mask = inputs != 0
        context_targets = inputs
        digit_indices = predictions
    context_f32 = context_mask.astype(jnp.float32)
    context_consistency = (
        jnp.sum(((predictions == context_targets) & context_mask).astype(jnp.float32))
        / jnp.maximum(jnp.sum(context_f32), 1.0)
    )
    if model.config.task_type != "sudoku":
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return context_consistency, zero
    if model.config.grid_height != 9 or model.config.grid_width != 9 or model.config.vocab_size != 9:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return context_consistency, zero

    one_hot = jax.nn.one_hot(digit_indices, 9)
    grid = one_hot.reshape(predictions.shape[0], 9, 9, 9)
    row_counts = jnp.sum(grid, axis=2)
    col_counts = jnp.sum(grid, axis=1)
    box_values = jnp.take(one_hot, model.box_indices, axis=1)
    box_counts = jnp.sum(box_values, axis=2)
    row_conflicts = jnp.sum(jnp.maximum(row_counts - 1.0, 0.0), axis=(1, 2))
    col_conflicts = jnp.sum(jnp.maximum(col_counts - 1.0, 0.0), axis=(1, 2))
    box_conflicts = jnp.sum(jnp.maximum(box_counts - 1.0, 0.0), axis=(1, 2))
    conflicts = jnp.mean(row_conflicts + col_conflicts + box_conflicts)
    return context_consistency, conflicts


def _bdr_region_masks(model: BeliefDynamicsReasoner, inputs: jax.Array, loss_mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    if model.config.task_type != "sudoku":
        zero_context = jnp.zeros_like(loss_mask, dtype=jnp.float32)
        return zero_context, (loss_mask > 0.0).astype(jnp.float32)
    input_context = model.context_mask(inputs)
    context_mask = input_context & (loss_mask > 0.0)
    query_mask = (~input_context) & (loss_mask > 0.0)
    return context_mask.astype(jnp.float32), query_mask.astype(jnp.float32)


def _masked_cell_accuracy(predictions: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
    correct = (predictions == targets).astype(jnp.float32) * mask.astype(jnp.float32)
    return jnp.sum(correct) / jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)


def _masked_probability(probability: jax.Array, mask: jax.Array) -> jax.Array:
    mask_f32 = mask.astype(jnp.float32)
    return jnp.sum(probability * mask_f32) / jnp.maximum(jnp.sum(mask_f32), 1.0)


def _masked_output_confidence(model: object, outputs: jax.Array, mask: jax.Array) -> jax.Array:
    confidence = jnp.max(output_probabilities(model, outputs), axis=-1)
    return _masked_probability(confidence, mask)


def _output_predictions_to_tokens(model: object, logits: jax.Array) -> jax.Array:
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def _normalized_step_loss_weights(configured: tuple[float, ...] | None, rollout_steps: int) -> jax.Array:
    if configured is None:
        weights = jnp.arange(1, rollout_steps + 1, dtype=jnp.float32)
    else:
        weights = jnp.asarray(configured, dtype=jnp.float32)
    return weights / jnp.maximum(jnp.sum(weights), 1e-6)


def _bdr_step_loss_weights(model: BeliefDynamicsReasoner, rollout_steps: int) -> jax.Array:
    progress = jnp.arange(1, rollout_steps + 1, dtype=jnp.float32) / float(rollout_steps)
    if model.bdr.step_loss_schedule == "quadratic":
        weights = jnp.square(progress)
    elif model.bdr.step_loss_schedule == "linear":
        weights = progress
    else:
        weights = jnp.ones((rollout_steps,), dtype=jnp.float32)
    return weights / jnp.maximum(jnp.sum(weights), 1e-6)


def _bdr_branch_diagnostics(model: BeliefDynamicsReasoner, diagnostics: dict[str, jax.Array]) -> dict[str, jax.Array]:
    update_rule = model.bdr.update_rule
    if update_rule == "proposal":
        return {
            "proposal_update_rms": diagnostics["proposal_update_rms"],
            "proposal_entropy": diagnostics["proposal_entropy"],
        }
    if update_rule == "free_velocity":
        return {
            "free_velocity_rms": diagnostics["free_velocity_rms"],
            "free_velocity_negative_rate": diagnostics["free_velocity_negative_rate"],
        }
    if update_rule == "energy_dist":
        return {
            "energy_update_rms": diagnostics["energy_update_rms"],
            "energy_value": diagnostics["energy_value"],
            "energy_distribution_step_rms": diagnostics["energy_distribution_step_rms"],
            "energy_entropy": diagnostics["energy_entropy"],
        }
    if update_rule == "energy_prob":
        return {
            "energy_update_rms": diagnostics["energy_update_rms"],
            "energy_value": diagnostics["energy_value"],
            "energy_grad_rms": diagnostics["energy_grad_rms"],
            "energy_probability_step_rms": diagnostics["logit_step_rms"],
        }
    raise ValueError(f"Unsupported BDR update_rule={update_rule!r}")


def _mean_bdr_branch_diagnostics(model: BeliefDynamicsReasoner, diagnostics: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {name: jnp.mean(value) for name, value in _bdr_branch_diagnostics(model, diagnostics).items()}


def _per_step_bdr_branch_diagnostics(model: BeliefDynamicsReasoner, diagnostics: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        f"per_step_{name}": value
        for name, value in _bdr_branch_diagnostics(model, diagnostics).items()
    }


def _bdr_attractor_metrics(model: BeliefDynamicsReasoner, losses: dict[str, jax.Array]) -> dict[str, jax.Array]:
    metrics: dict[str, jax.Array] = {}
    has_energy_rank = model.bdr.update_rule in ("energy_prob", "energy_dist")
    if float(model.bdr.wrong_attractor_rank_weight) != 0.0 and has_energy_rank:
        metrics["wrong_attractor_rank_loss"] = losses["wrong_attractor_rank_loss"]
        metrics["wrong_attractor_energy_gap"] = losses["wrong_attractor_energy_gap"]
        metrics["wrong_attractor_active_rate"] = losses["wrong_attractor_active_rate"]
    if float(model.bdr.wrong_attractor_direction_weight) != 0.0:
        metrics["wrong_attractor_direction_loss"] = losses["wrong_attractor_direction_loss"]
        metrics["wrong_attractor_direction_cosine"] = losses["wrong_attractor_direction_cosine"]
        metrics["wrong_attractor_active_rate"] = losses["wrong_attractor_active_rate"]
    if float(model.bdr.wrong_attractor_nonzero_weight) != 0.0:
        metrics["wrong_attractor_nonzero_loss"] = losses["wrong_attractor_nonzero_loss"]
        metrics["wrong_attractor_active_rate"] = losses["wrong_attractor_active_rate"]
    if float(model.bdr.corrupted_recovery_weight) != 0.0:
        metrics["corrupted_recovery_loss"] = losses["corrupted_recovery_loss"]
        metrics["corrupted_recovery_direction_cosine"] = losses["corrupted_recovery_direction_cosine"]
        if has_energy_rank:
            metrics["corrupted_recovery_rank_loss"] = losses["corrupted_recovery_rank_loss"]
            metrics["corrupted_recovery_energy_gap"] = losses["corrupted_recovery_energy_gap"]
    return metrics


def _recurrent_step_loss_weights(model: GridReasoningModel, rollout_steps: int) -> jax.Array:
    if isinstance(model, BeliefDynamicsReasoner):
        return _bdr_step_loss_weights(model, rollout_steps)
    recurrent_config = getattr(model, "trm", None) or getattr(model, "urm", None)
    return _normalized_step_loss_weights(getattr(recurrent_config, "step_loss_weights", None), rollout_steps)


def _selected_step(halt_logits: jax.Array) -> jax.Array:
    step_halted = halt_logits > 0.0
    step_halted = step_halted.at[-1, :].set(True)
    return jnp.argmax(step_halted.astype(jnp.int32), axis=0)


def loss_and_metrics(
    model: GridReasoningModel,
    batch: dict[str, jax.Array],
    train: bool,
    dropout_key: jax.Array | None,
    halt_loss_weight: float = 0.0,
    terminal_residual_weight: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if isinstance(model, (BeliefDynamicsReasoner, TinyRecursiveModel, UnifiedReasoningModel)):
        del terminal_residual_weight
        if train:
            raise ValueError("Recurrent model training uses ACT carry mode; dense unroll training has been removed")
        return recurrent_eval_loss_and_metrics(
            model,
            batch,
            halt_loss_weight,
            collect_diagnostics=True,
        )

    inputs = batch["inputs"]
    targets = batch["labels"]
    example_mask = _example_mask(batch, targets)
    loss_mask = _apply_example_mask(supervised_loss_mask(model, targets), example_mask)
    normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    metric_normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)

    model_key = dropout_key

    compute_terminal_residual = terminal_residual_weight != 0.0
    model_forward_kwargs = {}
    if isinstance(model, (TinyRecursiveModel, UnifiedReasoningModel)):
        model_forward_kwargs = {
            "puzzle_identifiers": batch["puzzle_identifiers"],
        }
        if isinstance(model, TinyRecursiveModel):
            model_forward_kwargs["compute_terminal_residual"] = compute_terminal_residual
    step_logits, diagnostics = model.forward_all_steps_with_diagnostics(
        inputs,
        train=train,
        dropout_key=model_key,
        **model_forward_kwargs,
    )
    effective_step_logits = step_logits
    step_targets = jnp.broadcast_to(targets[None, :, :], step_logits.shape[:-1])
    step_loss_mask = loss_mask[None, :, :]
    metric_step_loss_mask = loss_mask[None, :, :]
    token_loss = token_cross_entropy(model, effective_step_logits, step_targets)
    per_step_loss = jnp.sum(token_loss * step_loss_mask, axis=(1, 2)) / normalizer
    per_example_normalizer = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1.0)
    per_step_example_loss = jnp.sum(token_loss * step_loss_mask, axis=-1) / per_example_normalizer[None, :]

    rollout_steps = effective_step_logits.shape[0]
    step_weights = _recurrent_step_loss_weights(model, rollout_steps)
    lm_loss_value = jnp.sum(step_weights * per_step_loss)
    supervised_cells_per_example = jnp.sum(loss_mask, axis=-1)
    step_predictions = _output_predictions_to_tokens(model, effective_step_logits)
    step_correct = (step_predictions == step_targets).astype(jnp.float32) * metric_step_loss_mask
    step_correct_per_example = jnp.sum(step_correct, axis=-1)
    per_step_example_solved = jnp.where(
        supervised_cells_per_example[None, :] > 0,
        step_correct_per_example == supervised_cells_per_example[None, :],
        True,
    )
    halt_logits = diagnostics.get("halt_logits")
    halt_loss = jnp.asarray(0.0, dtype=jnp.float32)
    selected_logits = effective_step_logits[-1]
    selected_step = jnp.full((inputs.shape[0],), effective_step_logits.shape[0] - 1, dtype=jnp.int32)
    if halt_logits is not None and halt_loss_weight != 0.0:
        halt_targets = jax.lax.stop_gradient(per_step_example_solved.astype(jnp.float32))
        halt_loss_per_example = optax.sigmoid_binary_cross_entropy(halt_logits, halt_targets)
        halt_loss = jnp.sum(halt_loss_per_example * example_mask[None, :]) / jnp.maximum(
            jnp.sum(example_mask) * halt_logits.shape[0],
            1.0,
        )
        selected_step = _selected_step(halt_logits)
        gather_index = selected_step[None, :, None, None]
        selected_logits = jnp.take_along_axis(
            effective_step_logits,
            jnp.broadcast_to(gather_index, (1, inputs.shape[0], effective_step_logits.shape[2], effective_step_logits.shape[3])),
            axis=0,
        ).squeeze(0)
    terminal_residual = diagnostics.get(
        "terminal_belief_mse",
        diagnostics.get("terminal_belief_delta", jnp.asarray(0.0, dtype=jnp.float32)),
    )
    loss = (
        lm_loss_value
        + halt_loss_weight * halt_loss
        + terminal_residual_weight * terminal_residual
    )

    final_logits = effective_step_logits[-1]
    predictions = _output_predictions_to_tokens(model, final_logits)
    target_probability = token_target_probability(model, final_logits, targets)
    target_probability = jnp.sum(target_probability * loss_mask) / metric_normalizer
    correct = (predictions == targets).astype(jnp.float32) * loss_mask
    cell_accuracy = jnp.sum(correct) / metric_normalizer

    correct_per_example = jnp.sum(correct, axis=-1)
    exact_examples = jnp.where(
        supervised_cells_per_example > 0,
        correct_per_example == supervised_cells_per_example,
        True,
    )
    exact_f32 = exact_examples.astype(jnp.float32)
    exact_accuracy = _masked_example_mean(exact_f32, example_mask)
    exact_count = jnp.sum(exact_f32 * example_mask)
    if halt_loss_weight != 0.0:
        selected_token_loss = token_cross_entropy(model, selected_logits, targets)
        selected_ce_loss = jnp.sum(selected_token_loss * loss_mask) / normalizer
        selected_predictions = _output_predictions_to_tokens(model, selected_logits)
        selected_correct = (selected_predictions == targets).astype(jnp.float32) * loss_mask
        selected_accuracy = jnp.sum(selected_correct) / metric_normalizer
        selected_correct_per_example = jnp.sum(selected_correct, axis=-1)
        selected_exact_examples = jnp.where(
            supervised_cells_per_example > 0,
            selected_correct_per_example == supervised_cells_per_example,
            True,
        )
        selected_exact_accuracy = _masked_example_mean(selected_exact_examples.astype(jnp.float32), example_mask)
    else:
        selected_ce_loss = per_step_loss[-1]
        selected_accuracy = cell_accuracy
        selected_exact_accuracy = exact_accuracy
    oracle_step = jnp.argmin(per_step_example_loss, axis=0)

    metrics = {
        "loss": loss,
        "token_loss": lm_loss_value,
        "halt_loss": halt_loss,
        "final_token_loss": per_step_loss[-1],
        "final_target_probability": target_probability,
        "accuracy": cell_accuracy,
        "exact_accuracy": exact_accuracy,
        "exact_count": exact_count,
        "selected_token_loss": selected_ce_loss,
        "selected_accuracy": selected_accuracy,
        "selected_exact_accuracy": selected_exact_accuracy,
        "selected_step": _masked_example_mean(selected_step.astype(jnp.float32) + 1.0, example_mask),
        "oracle_step": _masked_example_mean(oracle_step.astype(jnp.float32) + 1.0, example_mask),
        "per_step_loss": per_step_loss,
        "step_loss_weights": step_weights,
        "per_step_hidden_delta": diagnostics["hidden_delta_mean"],
    }
    metrics.update(_maybe_path_metrics(model, predictions, targets, loss_mask))
    if halt_loss_weight != 0.0:
        halt_path_metrics = _maybe_path_metrics(model, selected_predictions, targets, loss_mask)
        metrics.update(
            {
                f"selected_{key}": value
                for key, value in halt_path_metrics.items()
                if key in ("path_precision", "path_recall", "path_f1")
            }
        )
    if halt_logits is not None and halt_loss_weight != 0.0:
        metrics["per_step_halt_probability"] = (
            jnp.sum(jax.nn.sigmoid(halt_logits) * example_mask[None, :], axis=1)
            / jnp.maximum(jnp.sum(example_mask), 1.0)
        )
    if "terminal_belief_delta" in diagnostics or terminal_residual_weight != 0.0:
        metrics["terminal_belief_delta"] = diagnostics.get("terminal_belief_delta", terminal_residual)
    if "terminal_belief_mse" in diagnostics or terminal_residual_weight != 0.0:
        metrics["terminal_belief_mse"] = terminal_residual
    if "rho_mean" in diagnostics:
        metrics["per_step_rho"] = diagnostics["rho_mean"]
    for key in (
        "unroll_steps",
        "terminal_belief_delta",
        "terminal_belief_mse",
        "belief_entropy",
        "belief_confidence",
        "halt_loss",
        "selected_token_loss",
        "selected_accuracy",
        "selected_exact_accuracy",
        "selected_step",
        "oracle_step",
    ):
        if key in diagnostics:
            metrics[key] = diagnostics[key]
    return loss, metrics


def act_loss_and_metrics(
    model: GridReasoningModel,
    carry: dict[str, jax.Array],
    batch: dict[str, jax.Array],
    train: bool,
    dropout_key: jax.Array | None,
    halt_loss_weight: float = 0.5,
    puzzle_embeddings: jax.Array | None = None,
) -> tuple[jax.Array, tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
    if dropout_key is None:
        dropout_key = jax.random.key(0)
    step_key, fixed_point_key, attractor_key = jax.random.split(dropout_key, 3)
    new_carry, logits, diagnostics = model.forward_act_step(
        carry,
        batch,
        train=train,
        dropout_key=step_key,
        puzzle_embeddings=puzzle_embeddings,
    )
    inputs = new_carry["current_inputs"]
    targets = new_carry["current_labels"]
    example_mask = _example_mask(batch, targets)
    loss_mask = _apply_example_mask(supervised_loss_mask(model, targets), example_mask)
    normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    metric_normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    per_example_normalizer = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1.0)
    token_loss = token_cross_entropy(model, logits, targets)
    per_example_loss = jnp.sum(token_loss * loss_mask, axis=-1) / per_example_normalizer
    lm_loss_value = jnp.mean(per_example_loss)

    predictions = _output_predictions_to_tokens(model, logits)
    correct = (predictions == targets).astype(jnp.float32) * loss_mask
    current_cell_accuracy = jnp.sum(correct) / metric_normalizer
    correct_per_example = jnp.sum(correct, axis=-1)
    supervised_cells_per_example = jnp.sum(loss_mask, axis=-1)
    exact_examples = jnp.where(
        supervised_cells_per_example > 0,
        correct_per_example == supervised_cells_per_example,
        True,
    )
    exact_targets = jax.lax.stop_gradient(exact_examples.astype(jnp.float32))
    current_exact_accuracy = jnp.mean(exact_targets)
    current_exact_count = jnp.sum(exact_targets)
    valid_halted = new_carry["halted"] & (supervised_cells_per_example > 0)
    valid_halted_f32 = valid_halted.astype(jnp.float32)
    valid_halted_count = jnp.sum(valid_halted_f32)
    valid_halted_normalizer = jnp.maximum(valid_halted_count, 1.0)
    per_example_accuracy = correct_per_example / jnp.maximum(supervised_cells_per_example, 1.0)
    cell_accuracy = jnp.sum(per_example_accuracy * valid_halted_f32) / valid_halted_normalizer
    exact_accuracy = jnp.sum(exact_targets * valid_halted_f32) / valid_halted_normalizer
    exact_count = jnp.sum(exact_targets * valid_halted_f32)
    q_halt_correct = ((diagnostics["halt_logits"] >= 0.0) == exact_examples) & valid_halted
    q_halt_accuracy = jnp.sum(q_halt_correct.astype(jnp.float32)) / valid_halted_normalizer
    steps = jnp.where(valid_halted_count > 0.0, diagnostics["act_step"], 0.0)

    halt_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(diagnostics["halt_logits"], exact_targets))
    loss = lm_loss_value + halt_loss_weight * halt_loss
    bdr_metrics: dict[str, jax.Array] = {}
    if isinstance(model, BeliefDynamicsReasoner):
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        puzzle_identifiers = new_carry.get("current_puzzle_identifiers")
        path_energy_loss = diagnostics["path_energy"].astype(jnp.float32)
        if float(model.bdr.fixed_point_update_weight) != 0.0:
            fixed_point_update_loss = model.fixed_point_update_loss(
                inputs,
                targets,
                loss_mask,
                puzzle_identifiers=puzzle_identifiers,
                train=train,
                dropout_key=fixed_point_key,
            )
        else:
            fixed_point_update_loss = zero
        use_attractor_losses = (
            float(model.bdr.wrong_attractor_rank_weight) != 0.0
            or float(model.bdr.wrong_attractor_direction_weight) != 0.0
            or float(model.bdr.wrong_attractor_nonzero_weight) != 0.0
            or float(model.bdr.corrupted_recovery_weight) != 0.0
        )
        if use_attractor_losses:
            attractor_losses = model.attractor_recovery_losses(
                inputs,
                targets,
                loss_mask,
                new_carry["z"],
                new_carry["hidden"],
                puzzle_identifiers=puzzle_identifiers,
                train=train,
                dropout_key=attractor_key,
            )
            attractor_metrics = _bdr_attractor_metrics(model, attractor_losses)
        else:
            attractor_losses = {
                "wrong_attractor_rank_loss": zero,
                "wrong_attractor_direction_loss": zero,
                "wrong_attractor_nonzero_loss": zero,
                "corrupted_recovery_loss": zero,
            }
            attractor_metrics = {}
        loss = (
            loss
            + float(model.bdr.path_energy_weight) * path_energy_loss
            + float(model.bdr.fixed_point_update_weight) * fixed_point_update_loss
            + float(model.bdr.wrong_attractor_rank_weight) * attractor_losses["wrong_attractor_rank_loss"]
            + float(model.bdr.wrong_attractor_direction_weight) * attractor_losses["wrong_attractor_direction_loss"]
            + float(model.bdr.wrong_attractor_nonzero_weight) * attractor_losses["wrong_attractor_nonzero_loss"]
            + float(model.bdr.corrupted_recovery_weight) * attractor_losses["corrupted_recovery_loss"]
        )
        context_mask, query_mask = _bdr_region_masks(model, inputs, loss_mask)
        query_accuracy = _masked_cell_accuracy(predictions, targets, query_mask)
        target_probability_cells = token_target_probability(model, logits, targets)
        bdr_metrics = {
            "path_energy_loss": path_energy_loss,
            "fixed_point_update_loss": fixed_point_update_loss,
            **attractor_metrics,
            "query_accuracy": query_accuracy,
            "query_target_probability": _masked_probability(target_probability_cells, query_mask),
            "distribution_tv_delta": diagnostics["distribution_tv_delta"],
            "path_energy": diagnostics["path_energy"],
            **_bdr_branch_diagnostics(model, diagnostics),
        }
        if model.config.task_type == "sudoku":
            bdr_metrics["context_accuracy"] = _masked_cell_accuracy(predictions, targets, context_mask)
            bdr_metrics["context_target_probability"] = _masked_probability(target_probability_cells, context_mask)
            context_consistency, conflicts = _sudoku_board_metrics(model, predictions, inputs)
            bdr_metrics["context_consistency"] = context_consistency
            bdr_metrics["conflicts"] = conflicts

    target_probability = token_target_probability(model, logits, targets)
    per_example_target_probability = (
        jnp.sum(target_probability * loss_mask, axis=-1)
        / jnp.maximum(supervised_cells_per_example, 1.0)
    )
    target_probability = (
        jnp.sum(per_example_target_probability * valid_halted_f32)
        / valid_halted_normalizer
    )
    metrics = {
        "loss": loss,
        "token_loss": lm_loss_value,
        "count": valid_halted_count,
        "accuracy": cell_accuracy,
        "exact_accuracy": exact_accuracy,
        "q_halt_accuracy": q_halt_accuracy,
        "steps": steps,
        "final_token_loss": lm_loss_value,
        "halted_target_probability": target_probability,
        "current_accuracy": current_cell_accuracy,
        "current_exact_accuracy": current_exact_accuracy,
        "current_exact_count": current_exact_count,
        "halt_loss": halt_loss,
        "act_step": diagnostics["act_step"],
        "halted_rate": diagnostics["halted_rate"],
        "reset_rate": diagnostics["reset_rate"],
    }
    metrics.update(bdr_metrics)
    metrics.update(_maybe_path_metrics(model, predictions, targets, loss_mask))
    return loss, (metrics, new_carry)


def recurrent_eval_loss_and_metrics(
    model: GridReasoningModel,
    batch: dict[str, jax.Array],
    halt_loss_weight: float = 0.5,
    *,
    collect_diagnostics: bool = False,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    inputs = batch["inputs"]
    targets = batch["labels"]
    example_mask = _example_mask(batch, targets)
    loss_mask = _apply_example_mask(supervised_loss_mask(model, targets), example_mask)
    normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    metric_normalizer = jnp.maximum(jnp.sum(loss_mask), 1.0)
    per_example_normalizer = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1.0)

    model_forward_kwargs = {"puzzle_identifiers": batch.get("puzzle_identifiers")}
    if isinstance(model, TinyRecursiveModel):
        model_forward_kwargs["compute_terminal_residual"] = False
        model_forward_kwargs["collect_diagnostics"] = collect_diagnostics
    elif isinstance(model, UnifiedReasoningModel):
        model_forward_kwargs["collect_diagnostics"] = collect_diagnostics
    step_logits, model_diagnostics = model.forward_all_steps_with_diagnostics(
        inputs,
        train=False,
        dropout_key=None,
        **model_forward_kwargs,
    )
    step_targets = jnp.broadcast_to(targets[None, :, :], step_logits.shape[:-1])
    step_loss_mask = loss_mask[None, :, :]
    metric_step_loss_mask = loss_mask[None, :, :]
    token_loss = token_cross_entropy(model, step_logits, step_targets)

    step_predictions = _output_predictions_to_tokens(model, step_logits)
    step_correct = (step_predictions == step_targets).astype(jnp.float32) * metric_step_loss_mask
    step_correct_per_example = jnp.sum(step_correct, axis=-1)
    supervised_cells_per_example = jnp.sum(loss_mask, axis=-1)
    per_step_example_solved = jnp.where(
        supervised_cells_per_example[None, :] > 0,
        step_correct_per_example == supervised_cells_per_example[None, :],
        True,
    )
    halt_logits = model_diagnostics["halt_logits"]
    selected_step = _selected_step(halt_logits)
    gather_index = selected_step[None, :, None, None]
    selected_logits = jnp.take_along_axis(
        step_logits,
        jnp.broadcast_to(gather_index, (1, inputs.shape[0], step_logits.shape[2], step_logits.shape[3])),
        axis=0,
    ).squeeze(0)
    selected_token_loss = token_cross_entropy(model, selected_logits, targets)
    selected_lm_loss_value = jnp.sum(selected_token_loss * loss_mask) / normalizer
    selected_exact_targets = jnp.take_along_axis(
        per_step_example_solved.astype(jnp.float32),
        selected_step[None, :],
        axis=0,
    ).squeeze(0)
    selected_exact_count = jnp.sum(selected_exact_targets * example_mask)
    halt_loss_per_example = optax.sigmoid_binary_cross_entropy(
        halt_logits,
        jax.lax.stop_gradient(per_step_example_solved.astype(jnp.float32)),
    )
    halt_loss = jnp.sum(halt_loss_per_example * example_mask[None, :]) / jnp.maximum(
        jnp.sum(example_mask) * halt_logits.shape[0],
        1.0,
    )
    final_logits = step_logits[-1]
    final_token_loss = token_cross_entropy(model, final_logits, targets)
    final_lm_loss_value = jnp.sum(final_token_loss * loss_mask) / normalizer
    loss = final_lm_loss_value + halt_loss_weight * halt_loss

    predictions = _output_predictions_to_tokens(model, selected_logits)
    target_probability = token_target_probability(model, selected_logits, targets)
    target_probability = jnp.sum(target_probability * loss_mask) / metric_normalizer
    correct = (predictions == targets).astype(jnp.float32) * loss_mask
    cell_accuracy = jnp.sum(correct) / metric_normalizer
    exact_accuracy = _masked_example_mean(selected_exact_targets, example_mask)
    selected_q_halt_logits = jnp.take_along_axis(halt_logits, selected_step[None, :], axis=0).squeeze(0)
    selected_q_halt_correct = (selected_q_halt_logits >= 0.0) == selected_exact_targets.astype(bool)
    q_halt_accuracy = _masked_example_mean(selected_q_halt_correct.astype(jnp.float32), example_mask)
    final_predictions = _output_predictions_to_tokens(model, final_logits)
    final_correct = (final_predictions == targets).astype(jnp.float32) * loss_mask
    final_cell_accuracy = jnp.sum(final_correct) / metric_normalizer
    final_correct_per_example = jnp.sum(final_correct, axis=-1)
    final_exact_examples = jnp.where(
        supervised_cells_per_example > 0,
        final_correct_per_example == supervised_cells_per_example,
        True,
    )
    final_exact_f32 = final_exact_examples.astype(jnp.float32)
    final_exact_count = jnp.sum(final_exact_f32 * example_mask)
    count = jnp.sum(example_mask)
    metrics = {
        "loss": loss,
        "token_loss": final_lm_loss_value,
        "count": count,
        "accuracy": final_cell_accuracy,
        "exact_accuracy": _masked_example_mean(final_exact_f32, example_mask),
        "q_halt_accuracy": q_halt_accuracy,
        "steps": jnp.asarray(step_logits.shape[0], dtype=jnp.float32),
        "halt_loss": halt_loss,
        "final_token_loss": final_lm_loss_value,
        "final_accuracy": final_cell_accuracy,
        "final_exact_accuracy": _masked_example_mean(final_exact_f32, example_mask),
        "halted_target_probability": target_probability,
        "exact_count": final_exact_count,
        "selected_token_loss": selected_lm_loss_value,
        "selected_accuracy": cell_accuracy,
        "selected_exact_accuracy": exact_accuracy,
        "selected_exact_count": selected_exact_count,
        "selected_step": _masked_example_mean(selected_step.astype(jnp.float32) + 1.0, example_mask),
        "final_exact_count": final_exact_count,
        "unroll_steps": jnp.asarray(step_logits.shape[0], dtype=jnp.int32),
    }
    if collect_diagnostics:
        per_step_loss = jnp.sum(token_loss * step_loss_mask, axis=(1, 2)) / normalizer
        per_step_example_loss = jnp.sum(token_loss * step_loss_mask, axis=-1) / per_example_normalizer[None, :]
        per_step_accuracy = jnp.sum(step_correct, axis=(1, 2)) / metric_normalizer
        oracle_step = jnp.argmin(per_step_example_loss, axis=0)
        metrics.update(
            {
                "oracle_step": _masked_example_mean(oracle_step.astype(jnp.float32) + 1.0, example_mask),
                "mean_token_loss": jnp.mean(per_step_loss),
                "step_loss_weights": _recurrent_step_loss_weights(model, step_logits.shape[0]),
                "per_step_loss": per_step_loss,
                "per_step_accuracy": per_step_accuracy,
                "per_step_hidden_delta": model_diagnostics["hidden_delta_mean"],
                "per_step_halt_probability": (
                    jnp.sum(jax.nn.sigmoid(halt_logits) * example_mask[None, :], axis=1)
                    / jnp.maximum(count, 1.0)
                ),
            }
        )
    selected_path_metrics = _maybe_path_metrics(model, predictions, targets, loss_mask)
    metrics.update(selected_path_metrics)
    metrics.update(
        {
            f"selected_{key}": value
            for key, value in selected_path_metrics.items()
            if key in ("path_precision", "path_recall", "path_f1")
        }
    )
    final_path_metrics = _maybe_path_metrics(model, final_predictions, targets, loss_mask)
    metrics.update(
        {
            f"final_{key}": value
            for key, value in final_path_metrics.items()
            if key in ("path_precision", "path_recall", "path_f1")
        }
    )
    if isinstance(model, BeliefDynamicsReasoner):
        context_mask, query_mask = _bdr_region_masks(model, inputs, loss_mask)
        final_target_probability_cells = token_target_probability(model, final_logits, targets)
        metrics.update(
            {
                "query_accuracy": _masked_cell_accuracy(final_predictions, targets, query_mask),
                "query_target_probability": _masked_probability(final_target_probability_cells, query_mask),
                "distribution_tv_delta": jnp.mean(model_diagnostics["distribution_tv_delta"]),
                "path_energy": jnp.mean(model_diagnostics["path_energy"]),
                "path_energy_loss": jnp.mean(model_diagnostics["path_energy"]),
                **_mean_bdr_branch_diagnostics(model, model_diagnostics),
            }
        )
        if model.config.task_type == "sudoku":
            context_consistency, conflicts = _sudoku_board_metrics(model, final_predictions, inputs)
            metrics.update(
                {
                    "context_accuracy": _masked_cell_accuracy(final_predictions, targets, context_mask),
                    "context_target_probability": _masked_probability(final_target_probability_cells, context_mask),
                    "context_consistency": context_consistency,
                    "conflicts": conflicts,
                }
            )
        if collect_diagnostics:
            metrics.update(
                {
                    "per_step_q_top1_probability": model_diagnostics["q_top1_probability"],
                    "per_step_distribution_tv_delta": model_diagnostics["distribution_tv_delta"],
                    "per_step_path_energy": model_diagnostics["path_energy"],
                    **_per_step_bdr_branch_diagnostics(model, model_diagnostics),
                }
            )
    return loss, metrics
