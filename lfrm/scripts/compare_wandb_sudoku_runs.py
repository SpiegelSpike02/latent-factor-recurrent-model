from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import wandb


DEFAULT_ENTITY = "gjia0011-"
DEFAULT_PROJECT = "latent-factor-recurrent-model"
DEFAULT_OUT = Path("outputs") / "wandb_sudoku_run_comparison.csv"
DEFAULT_JSONL = Path("outputs") / "wandb_sudoku_run_comparison.jsonl"

TASK_CONFIG_PATHS = (
    "task.type",
    "task_type",
    "data.task",
    "dataset.task",
)

ARCH_CONFIG_PATHS = (
    "model.model_type",
    "model_type",
    "model.name",
    "architecture",
    "arch",
)

CONFIG_SUMMARY_PATHS = (
    "task.type",
    "task_type",
    "model.model_type",
    "model_type",
    "model.d_model",
    "d_model",
    "model.loss_type",
    "loss_type",
    "train.train_mode",
    "train_mode",
    "train.batch_size",
    "batch_size",
    "optimizer.learning_rate",
    "learning_rate",
    "model.bdr.update_rule",
    "model.brc.update_rule",
    "model.bdr.commit_steps",
    "model.brc.commit_steps",
    "model.bdr.refine_steps",
    "model.brc.refine_steps",
    "model.bdr.block_depth",
    "model.brc.block_depth",
    "model.H_cycles",
    "model.L_cycles",
    "model.H_layers",
    "model.L_layers",
)

ACCURACY_ALIASES = (
    "eval/ema/exact_accuracy",
    "eval/exact_accuracy",
    "eval/accuracy",
    "train/exact_accuracy",
    "train/query_accuracy",
    "query_accuracy",
    "exact_accuracy",
    "accuracy",
    "val/accuracy",
    "val/exact_accuracy",
    "valid/accuracy",
    "validation/accuracy",
)

LOWER_IS_BETTER_ALIASES = (
    "train/conflicts",
    "eval/conflicts",
    "eval/ema/conflicts",
    "conflicts",
    "train/loss",
    "eval/loss",
    "loss",
)

EXCLUDED_ACCURACY_FRAGMENTS = (
    "context",
    "clue",
    "given",
    "halt",
    "teacher",
)

FIELDNAMES = (
    "rank",
    "run_id",
    "name",
    "state",
    "created_at",
    "best_metric",
    "best_value",
    "best_step",
    "steps_to_50",
    "steps_to_80",
    "steps_to_95",
    "arch_key",
    "git_commit",
    "url",
    "config_summary_json",
    "secondary_metrics_json",
)


def scalar(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    try:
        number = float(value)
    except Exception:
        return str(value)
    return number if math.isfinite(number) else None


def config_get(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def config_get_any(config: dict[str, Any], paths: Iterable[str]) -> Any:
    for path in paths:
        value = config_get(config, path)
        if value is not None:
            return value
    return None


def flatten_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for path in CONFIG_SUMMARY_PATHS:
        value = config_get(config, path)
        if value is not None:
            summary[path] = scalar(value)
    return summary


def summary_dict(run: Any) -> dict[str, Any]:
    raw = getattr(run.summary, "_json_dict", None)
    if isinstance(raw, dict):
        return raw
    return dict(run.summary)


def metric_candidates(run: Any) -> list[str]:
    run_summary = summary_dict(run)
    keys: set[str] = set()
    for key in summary_dict(run):
        lowered = key.lower()
        if any(fragment in lowered for fragment in EXCLUDED_ACCURACY_FRAGMENTS):
            continue
        if "accuracy" in lowered or lowered.endswith("/acc") or lowered == "acc":
            keys.add(key)
        if "conflict" in lowered or lowered.endswith("/loss") or lowered == "loss":
            keys.add(key)
    for key in (*ACCURACY_ALIASES, *LOWER_IS_BETTER_ALIASES):
        if key in run_summary:
            keys.add(key)
    return sorted(keys)


def is_sudoku_run(run: Any, config: dict[str, Any], *, include_name_fallback: bool) -> bool:
    task = config_get_any(config, TASK_CONFIG_PATHS)
    if isinstance(task, str):
        return task.lower() == "sudoku"
    dataset_path = config_get(config, "data.dataset_path") or config_get(config, "dataset_path")
    if isinstance(dataset_path, str) and "sudoku" in dataset_path.lower():
        return True
    return include_name_fallback and "sudoku" in (run.name or "").lower()


def metric_priority(metric: str) -> tuple[int, str]:
    try:
        return (ACCURACY_ALIASES.index(metric), metric)
    except ValueError:
        pass
    lowered = metric.lower()
    if "exact" in lowered:
        return (20, metric)
    if "query" in lowered:
        return (30, metric)
    if metric.startswith("eval/"):
        return (40, metric)
    if metric.startswith("train/"):
        return (50, metric)
    return (60, metric)


def architecture_key(config: dict[str, Any]) -> str:
    arch = config_get_any(config, ARCH_CONFIG_PATHS) or "unknown"
    loss = config_get(config, "model.loss_type") or config_get(config, "loss_type")
    update_rule = config_get(config, "model.bdr.update_rule") or config_get(config, "model.brc.update_rule")
    train_mode = config_get(config, "train.train_mode") or config_get(config, "train_mode")
    pieces = [str(arch)]
    for value in (loss, update_rule, train_mode):
        if value is not None:
            pieces.append(str(value))
    return ":".join(pieces)


def get_step(row: dict[str, Any]) -> int | None:
    value = row.get("_step")
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def update_thresholds(thresholds: dict[float, int | None], value: float, step: int) -> None:
    for threshold, current_step in thresholds.items():
        if current_step is None and value >= threshold:
            thresholds[threshold] = step


def update_from_metric_value(
    *,
    metric: str,
    value: Any,
    step: int,
    best: dict[str, Any] | None,
    thresholds: dict[float, int | None],
    secondary: dict[str, Any],
) -> dict[str, Any] | None:
    number = scalar(value)
    if not isinstance(number, (int, float)):
        return best
    if metric in LOWER_IS_BETTER_ALIASES or "conflict" in metric.lower() or metric.endswith("/loss"):
        previous = secondary.get(metric)
        if previous is None or number < previous["value"]:
            secondary[metric] = {"value": number, "step": step}
        return best
    if metric in ACCURACY_ALIASES or "accuracy" in metric.lower() or metric.endswith("/acc"):
        priority = metric_priority(metric)
        candidate = {"metric": metric, "value": float(number), "step": step, "priority": priority}
        if best is None:
            best = candidate
        elif candidate["value"] > best["value"]:
            best = candidate
        elif candidate["value"] == best["value"] and candidate["step"] < best["step"]:
            best = candidate
        elif candidate["value"] == best["value"] and candidate["step"] == best["step"] and priority < best["priority"]:
            best = candidate
        update_thresholds(thresholds, float(number), step)
    return best


def collect_run_result(
    run: Any,
    *,
    history_mode: str,
    history_samples: int,
    scan_page_size: int,
    max_history_rows: int | None,
) -> dict[str, Any]:
    config = dict(run.config)
    metadata = dict(run.metadata or {})
    candidates = metric_candidates(run)
    best: dict[str, Any] | None = None
    thresholds: dict[float, int | None] = {0.50: None, 0.80: None, 0.95: None}
    secondary: dict[str, Any] = {}
    if history_mode == "sampled":
        rows = run.history(keys=["_step", *candidates], samples=history_samples, pandas=False)
        for row in rows:
            step = get_step(row)
            if step is None:
                continue
            for metric in candidates:
                if metric in row:
                    best = update_from_metric_value(
                        metric=metric,
                        value=row.get(metric),
                        step=step,
                        best=best,
                        thresholds=thresholds,
                        secondary=secondary,
                    )
    else:
        for metric in candidates:
            rows_seen = 0
            for row in run.scan_history(keys=["_step", metric], page_size=scan_page_size):
                rows_seen += 1
                step = get_step(row)
                if step is None or metric not in row:
                    continue
                best = update_from_metric_value(
                    metric=metric,
                    value=row.get(metric),
                    step=step,
                    best=best,
                    thresholds=thresholds,
                    secondary=secondary,
                )
                if max_history_rows is not None and rows_seen >= max_history_rows:
                    break
    summary = summary_dict(run)
    if best is None:
        for metric in candidates:
            if metric not in summary:
                continue
            value = scalar(summary.get(metric))
            if isinstance(value, (int, float)):
                best = {"metric": metric, "value": float(value), "step": None, "priority": metric_priority(metric)}
                break
    return {
        "run_id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "best_metric": None if best is None else best["metric"],
        "best_value": None if best is None else best["value"],
        "best_step": None if best is None else best["step"],
        "steps_to_50": thresholds[0.50],
        "steps_to_80": thresholds[0.80],
        "steps_to_95": thresholds[0.95],
        "arch_key": architecture_key(config),
        "git_commit": metadata.get("git", {}).get("commit") or metadata.get("git_commit"),
        "url": run.url,
        "config_summary": flatten_config_summary(config),
        "secondary_metrics": secondary,
    }


def sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
    best_value = row.get("best_value")
    best_step = row.get("best_step")
    return (
        -float(best_value) if isinstance(best_value, (int, float)) else float("inf"),
        int(best_step) if isinstance(best_step, int) else 10**18,
        str(row.get("run_id")),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "run_id": row["run_id"],
                    "name": row["name"],
                    "state": row["state"],
                    "created_at": row["created_at"],
                    "best_metric": row["best_metric"],
                    "best_value": row["best_value"],
                    "best_step": row["best_step"],
                    "steps_to_50": row["steps_to_50"],
                    "steps_to_80": row["steps_to_80"],
                    "steps_to_95": row["steps_to_95"],
                    "arch_key": row["arch_key"],
                    "git_commit": row["git_commit"],
                    "url": row["url"],
                    "config_summary_json": json.dumps(row["config_summary"], sort_keys=True),
                    "secondary_metrics_json": json.dumps(row["secondary_metrics"], sort_keys=True),
                }
            )
    tmp.replace(path)


def append_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream W&B Sudoku runs and rank architectures by accuracy per step.")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--runs-per-page", type=int, default=int(os.environ.get("WANDB_RUNS_PER_PAGE", "20")))
    parser.add_argument("--history-mode", choices=("sampled", "exact"), default="sampled")
    parser.add_argument("--history-samples", type=int, default=int(os.environ.get("WANDB_HISTORY_SAMPLES", "1000")))
    parser.add_argument("--scan-page-size", type=int, default=int(os.environ.get("WANDB_SCAN_PAGE_SIZE", "500")))
    parser.add_argument("--max-history-rows", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many Sudoku runs.")
    parser.add_argument("--include-name-fallback", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    api = wandb.Api(timeout=120)
    rows: list[dict[str, Any]] = []
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_jsonl = args.jsonl.with_suffix(args.jsonl.suffix + ".tmp")
    with tmp_jsonl.open("w", encoding="utf-8") as jsonl_handle:
        for run in api.runs(f"{args.entity}/{args.project}", order="-created_at", per_page=args.runs_per_page):
            config = dict(run.config)
            if not is_sudoku_run(run, config, include_name_fallback=args.include_name_fallback):
                continue
            row = collect_run_result(
                run,
                history_mode=args.history_mode,
                history_samples=args.history_samples,
                scan_page_size=args.scan_page_size,
                max_history_rows=args.max_history_rows,
            )
            rows.append(row)
            append_jsonl(jsonl_handle, row)
            best = row["best_value"]
            best_text = "--" if best is None else f"{best:.4f}"
            print(f"{len(rows):4d} {run.id} {best_text} step={row['best_step']} {row['arch_key']} {run.name}", flush=True)
            if args.limit is not None and len(rows) >= args.limit:
                break
    tmp_jsonl.replace(args.jsonl)
    rows.sort(key=sort_key)
    write_csv(args.out, rows)
    print(f"wrote {args.out} with {len(rows)} sudoku runs")
    print(f"wrote {args.jsonl}")


if __name__ == "__main__":
    main()
