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
    "lm_loss",
    "q_halt_loss",
    "accuracy",
    "context_accuracy",
    "query_accuracy",
    "exact_accuracy",
    "q_halt_accuracy",
    "steps",
    "count",
    "final_lm_loss",
    "final_accuracy",
    "final_exact_accuracy",
    "mean_lm_loss",
    "current_accuracy",
    "current_context_accuracy",
    "current_query_accuracy",
    "current_exact_accuracy",
    "halted_target_probability",
    "final_target_probability",
    "context_target_probability",
    "query_target_probability",
    "unroll_steps",
    "context_consistency",
    "invalid_rate",
    "conflicts",
    "diffusion_filled_ratio",
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
    "current_exact_count",
    "selected_exact_count",
    "final_exact_count",
    "path_precision",
    "path_recall",
    "path_f1",
    "path_positive_rate",
    "target_path_rate",
    "selected_path_precision",
    "selected_path_recall",
    "selected_path_f1",
    "final_path_precision",
    "final_path_recall",
    "final_path_f1",
)
WANDB_HISTORY_EXCLUDED_SCALAR_METRICS: set[str] = set()
TERMINAL_DIAGNOSTIC_METRICS = (
    "terminal_belief_delta",
    "terminal_belief_mse",
)
INTEGER_SCALAR_METRICS = {
    "unroll_steps",
    "count",
    "exact_count",
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
}
BRC_CONSOLE_GROUPS = (
    (
        "objective",
        (
            "loss",
            "lm_loss",
            "final_lm_loss",
            "mean_lm_loss",
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
        "current",
        (
            "current_accuracy",
            "current_query_accuracy",
            "current_context_accuracy",
            "current_exact_accuracy",
            "reset_rate",
        ),
    ),
    (
        "sudoku",
        (
            "context_consistency",
            "invalid_rate",
            "conflicts",
        ),
    ),
    (
        "dynamics",
        (
            "diffusion_filled_ratio",
            "halt_loss",
            "selected_step",
            "selected_accuracy",
            "selected_exact_accuracy",
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
    names = list(CORE_SCALAR_METRICS)
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
