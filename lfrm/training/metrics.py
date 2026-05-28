from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def flatten_step_metrics(prefix: str, values: list[float]) -> dict[str, float]:
    return {
        f"{prefix}/step_{index + 1:02d}": float(value)
        for index, value in enumerate(values)
    }


def format_step_summary(name: str, values: list[float]) -> str:
    if not values:
        return f"{name}=[]"
    middle_index = len(values) // 2
    return (
        f"{name}="
        f"first:{values[0]:.4f},"
        f"mid:{values[middle_index]:.4f},"
        f"last:{values[-1]:.4f}"
    )


def path_metrics(predictions: jax.Array, targets: jax.Array, loss_mask: jax.Array) -> dict[str, jax.Array]:
    mask = loss_mask.astype(bool)
    predicted_path = (predictions == 5) & mask
    target_path = (targets == 5) & mask
    true_positive = jnp.sum((predicted_path & target_path).astype(jnp.float32))
    predicted_positive = jnp.sum(predicted_path.astype(jnp.float32))
    target_positive = jnp.sum(target_path.astype(jnp.float32))
    mask_total = jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)
    precision = true_positive / jnp.maximum(predicted_positive, 1.0)
    recall = true_positive / jnp.maximum(target_positive, 1.0)
    f1 = jnp.where(
        precision + recall > 0.0,
        2.0 * precision * recall / (precision + recall),
        0.0,
    )
    return {
        "path_precision": precision,
        "path_recall": recall,
        "path_f1": f1,
        "path_positive_rate": predicted_positive / mask_total,
        "target_path_rate": target_positive / mask_total,
    }


def maybe_path_metrics(
    *,
    task_type: str,
    predictions: jax.Array,
    targets: jax.Array,
    loss_mask: jax.Array,
) -> dict[str, jax.Array]:
    if task_type != "maze":
        return {}
    return path_metrics(predictions, targets, loss_mask)


CORE_SCALAR_METRICS = (
    "loss",
    "ce_loss",
    "final_ce_loss",
    "mean_ce_loss",
    "active_ce_loss",
    "lm_loss",
    "q_halt_loss",
    "accuracy",
    "context_accuracy",
    "query_accuracy",
    "exact_accuracy",
    "q_halt_accuracy",
    "steps",
    "count",
    "completed_count",
    "final_lm_loss",
    "final_accuracy",
    "final_exact_accuracy",
    "mean_lm_loss",
    "active_lm_loss",
    "completed_accuracy",
    "completed_context_accuracy",
    "completed_query_accuracy",
    "completed_exact_accuracy",
    "completed_target_probability",
    "completed_context_target_probability",
    "completed_query_target_probability",
    "current_accuracy",
    "current_context_accuracy",
    "current_query_accuracy",
    "current_exact_accuracy",
    "active_accuracy",
    "active_context_accuracy",
    "active_query_accuracy",
    "active_exact_accuracy",
    "active_target_probability",
    "active_context_target_probability",
    "active_query_target_probability",
    "halted_target_probability",
    "final_target_probability",
    "context_target_probability",
    "query_target_probability",
    "unroll_steps",
    "context_consistency",
    "conflicts",
    "active_context_consistency",
    "active_conflicts",
    "q_top1_probability",
    "update_step_size",
    "update_clip_scale",
    "update_rms",
    "velocity_rms",
    "velocity_clip_scale",
    "energy_update_rms",
    "energy_value",
    "energy_grad_rms",
    "logit_step_rms",
    "distribution_tv_delta",
    "path_energy",
    "path_energy_loss",
    "step_loss_weights",
    "halt_loss",
    "selected_lm_loss",
    "selected_accuracy",
    "selected_exact_accuracy",
    "selected_step",
    "oracle_step",
    "act_step",
    "halted_rate",
    "reset_rate",
    "exact_count",
    "completed_exact_count",
    "current_exact_count",
    "selected_exact_count",
    "final_exact_count",
    "path_precision",
    "path_recall",
    "path_f1",
    "path_positive_rate",
    "target_path_rate",
    "active_path_precision",
    "active_path_recall",
    "active_path_f1",
    "active_path_positive_rate",
    "active_target_path_rate",
    "selected_path_precision",
    "selected_path_recall",
    "selected_path_f1",
    "final_path_precision",
    "final_path_recall",
    "final_path_f1",
)
BRC_SCALAR_METRICS = (
    "loss",
    "ce_loss",
    "final_ce_loss",
    "mean_ce_loss",
    "accuracy",
    "context_accuracy",
    "query_accuracy",
    "exact_accuracy",
    "final_target_probability",
    "context_target_probability",
    "query_target_probability",
    "oracle_step",
    "unroll_steps",
    "exact_count",
    "context_consistency",
    "conflicts",
    "path_precision",
    "path_recall",
    "path_f1",
    "path_positive_rate",
    "target_path_rate",
    "q_top1_probability",
    "update_step_size",
    "update_clip_scale",
    "update_rms",
    "velocity_rms",
    "velocity_clip_scale",
    "energy_update_rms",
    "energy_value",
    "energy_grad_rms",
    "logit_step_rms",
    "distribution_tv_delta",
    "path_energy",
    "path_energy_loss",
    "fixed_point_update_loss",
    "wrong_attractor_rank_loss",
    "wrong_attractor_direction_loss",
    "wrong_attractor_nonzero_loss",
    "wrong_attractor_active_rate",
    "wrong_attractor_direction_cosine",
    "wrong_attractor_energy_gap",
    "corrupted_recovery_loss",
    "corrupted_recovery_rank_loss",
    "corrupted_recovery_direction_cosine",
    "corrupted_recovery_energy_gap",
    "carry_step",
    "reset_rate",
    "max_step_reset_rate",
    "early_stop_rate",
    "stability_rate",
    "stable_steps",
    "early_stop_update_rms",
    "early_stop_distribution_delta",
    "early_stop_flip_rate",
    "early_stop_margin_min",
    "early_stop_constraint_rate",
)
WANDB_HISTORY_EXCLUDED_SCALAR_METRICS: set[str] = set()
TERMINAL_DIAGNOSTIC_METRICS = (
    "terminal_belief_delta",
    "terminal_belief_mse",
)
INTEGER_SCALAR_METRICS = {
    "unroll_steps",
    "count",
    "completed_count",
    "exact_count",
    "completed_exact_count",
    "current_exact_count",
    "selected_exact_count",
    "final_exact_count",
}
LEGACY_METRIC_NAMES = {
    "blank_ce_loss",
    "final_blank_ce_loss",
    "mean_blank_ce_loss",
    "step_weighted_ce_loss",
    "blank_cell_accuracy",
    "solved_rate",
    "solved_count",
    "target_probability",
    "current_target_probability",
    "current_blank_cell_accuracy",
    "current_solved_rate",
    "current_solved_count",
    "halted_count",
    "halt_selected_blank_ce_loss",
    "halt_selected_blank_cell_accuracy",
    "halt_selected_solved_rate",
    "halt_selected_solved_count",
    "halt_selected_step",
    "halt_selected_path_precision",
    "halt_selected_path_recall",
    "halt_selected_path_f1",
    "final_blank_cell_accuracy",
    "final_solved_rate",
    "final_solved_count",
    "invalid_board_rate",
    "conflict_count",
    "q_confidence",
    "distribution_delta",
}
BRC_CONSOLE_GROUPS = (
    (
        "objective",
        (
            "loss",
            "ce_loss",
            "final_ce_loss",
            "mean_ce_loss",
            "fixed_point_update_loss",
            "wrong_attractor_rank_loss",
            "wrong_attractor_direction_loss",
            "wrong_attractor_nonzero_loss",
            "corrupted_recovery_loss",
            "corrupted_recovery_rank_loss",
        ),
    ),
    (
        "solve",
        (
            "accuracy",
            "query_accuracy",
            "context_accuracy",
            "exact_accuracy",
            "final_target_probability",
            "query_target_probability",
            "exact_count",
        ),
    ),
    (
        "early_stop",
        (
            "carry_step",
            "reset_rate",
            "max_step_reset_rate",
            "early_stop_rate",
            "stability_rate",
            "stable_steps",
            "early_stop_update_rms",
            "early_stop_distribution_delta",
            "early_stop_flip_rate",
            "early_stop_margin_min",
            "early_stop_constraint_rate",
        ),
    ),
    (
        "sudoku",
        (
            "context_consistency",
            "conflicts",
        ),
    ),
    (
        "path",
        (
            "path_precision",
            "path_recall",
            "path_f1",
            "path_positive_rate",
            "target_path_rate",
        ),
    ),
    (
        "dynamics",
        (
            "q_top1_probability",
            "update_step_size",
            "update_clip_scale",
            "update_rms",
            "velocity_rms",
            "velocity_clip_scale",
            "energy_update_rms",
            "energy_value",
            "energy_grad_rms",
            "logit_step_rms",
            "distribution_tv_delta",
            "path_energy",
            "path_energy_loss",
            "wrong_attractor_active_rate",
            "wrong_attractor_direction_cosine",
            "wrong_attractor_energy_gap",
            "corrupted_recovery_direction_cosine",
            "corrupted_recovery_energy_gap",
        ),
    ),
)
TRM_CONSOLE_GROUPS = (
    ("objective", ("loss", "lm_loss", "q_halt_loss")),
    ("official", ("accuracy", "exact_accuracy", "q_halt_accuracy", "steps", "count")),
    ("current", ("current_accuracy", "current_exact_accuracy", "halted_target_probability", "reset_rate")),
    ("final", ("final_lm_loss", "final_accuracy", "final_exact_accuracy")),
    (
        "path",
        (
            "path_precision",
            "path_recall",
            "path_f1",
            "path_positive_rate",
            "target_path_rate",
            "final_path_precision",
            "final_path_recall",
            "final_path_f1",
        ),
    ),
    ("halt", ("selected_step", "oracle_step", "act_step", "halted_rate", "selected_accuracy", "selected_exact_accuracy")),
    ("dynamics", ("unroll_steps", "terminal_belief_delta", "terminal_belief_mse")),
)
METRIC_GROUPS = BRC_CONSOLE_GROUPS
CONSOLE_GROUPS_BY_MODEL = {
    "brc": BRC_CONSOLE_GROUPS,
    "trm": TRM_CONSOLE_GROUPS,
    "urm": TRM_CONSOLE_GROUPS,
}


def assert_no_legacy_metrics(metrics: dict[str, Any]) -> None:
    legacy = sorted(set(metrics) & LEGACY_METRIC_NAMES)
    if legacy:
        raise AssertionError(f"Legacy metric keys are no longer allowed: {legacy}")


def format_scalar_metric(name: str, value: Any) -> str:
    array = np.asarray(value)
    scalar = float(array)
    is_integer_like = np.isfinite(scalar) and np.isclose(scalar, round(scalar), atol=1e-6)
    is_integer_dtype = np.issubdtype(array.dtype, np.integer)
    if is_integer_dtype or name in INTEGER_SCALAR_METRICS or name.endswith(("_count", "_iters", "_steps")):
        if is_integer_like:
            return str(int(round(scalar)))
    if scalar != 0.0 and abs(scalar) < 1e-3:
        return f"{scalar:.2e}"
    return f"{scalar:.4f}"


def metric_display_name(name: str) -> str:
    return name


def scalar_metric_names(config) -> tuple[str, ...]:
    names = list(BRC_SCALAR_METRICS if config.model.model_type == "brc" else CORE_SCALAR_METRICS)
    if config.model.model_type == "brc" and config.model.task_type != "sudoku":
        names = [
            name for name in names
            if "context_" not in name
            and name not in ("context_accuracy", "context_target_probability", "context_consistency", "conflicts")
        ]
    if config.train.terminal_residual_weight != 0.0:
        names.extend(TERMINAL_DIAGNOSTIC_METRICS)
    return tuple(names)


def optional_scalar_log(
    prefix: str,
    metrics: dict[str, Any],
    names: tuple[str, ...],
    *,
    exclude_history: set[str] | None = None,
) -> dict[str, float]:
    assert_no_legacy_metrics(metrics)
    log: dict[str, float] = {}
    excluded = exclude_history or set()
    for name in names:
        if name in excluded:
            continue
        if name in metrics:
            value = metrics[name]
            if np.ndim(np.asarray(value)) == 0:
                log[f"{prefix}/{name}"] = float(value)
    return log


def optional_summary_log(prefix: str, metrics: dict[str, Any], names: set[str]) -> dict[str, float]:
    assert_no_legacy_metrics(metrics)
    log: dict[str, float] = {}
    for name in names:
        if name in metrics:
            value = metrics[name]
            if np.ndim(np.asarray(value)) == 0:
                log[f"{prefix}/{name}"] = float(value)
    return log


def grouped_scalar_summary(metrics: dict[str, Any], names: tuple[str, ...], model_type: str | None = None) -> str:
    assert_no_legacy_metrics(metrics)
    allowed = set(names)
    groups = CONSOLE_GROUPS_BY_MODEL.get(model_type or "", METRIC_GROUPS)
    lines: list[str] = []
    emitted: set[str] = set()
    for group_name, group_metrics in groups:
        parts = []
        for name in group_metrics:
            if name not in allowed or name not in metrics:
                continue
            value = metrics[name]
            if np.ndim(np.asarray(value)) != 0:
                continue
            parts.append(f"{metric_display_name(name)}={format_scalar_metric(name, value)}")
            emitted.add(name)
        if parts:
            lines.append(f"  {group_name}: " + " ".join(parts))

    return "\n".join(lines)
