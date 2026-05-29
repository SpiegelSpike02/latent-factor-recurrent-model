from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import wandb


DEFAULT_ENTITY = "gjia0011-"
DEFAULT_PROJECT = "latent-factor-recurrent-model"

METRIC_RENAMES = {
    "ce_loss": "token_loss",
    "lm_loss": "token_loss",
    "final_ce_loss": "final_token_loss",
    "final_lm_loss": "final_token_loss",
    "mean_ce_loss": "mean_token_loss",
    "mean_lm_loss": "mean_token_loss",
    "selected_lm_loss": "selected_token_loss",
    "q_halt_loss": "halt_loss",
}


def canonical_metric_key(key: str) -> str:
    parts = key.split("/")
    leaf = parts[-1]
    if leaf not in METRIC_RENAMES:
        return key
    parts[-1] = METRIC_RENAMES[leaf]
    return "/".join(parts)


def scalar_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    try:
        return float(value)
    except Exception:
        return None


def transformed_history_rows(run: Any, *, history_samples: int) -> tuple[OrderedDict[int, dict[str, Any]], dict[str, int]]:
    rows_by_step: OrderedDict[int, dict[str, Any]] = OrderedDict()
    stats = {
        "source_rows": 0,
        "logged_rows": 0,
        "renamed_values": 0,
        "dropped_internal_values": 0,
        "dropped_none_values": 0,
        "collisions_kept_existing": 0,
    }
    for row in run.history(samples=history_samples, pandas=False):
        stats["source_rows"] += 1
        if "_step" not in row or row["_step"] is None:
            continue
        step = int(row["_step"])
        output = rows_by_step.setdefault(step, {})
        for key, value in row.items():
            if key == "_step":
                continue
            if key.startswith("_"):
                stats["dropped_internal_values"] += 1
                continue
            value = scalar_or_none(value)
            if value is None:
                stats["dropped_none_values"] += 1
                continue
            new_key = canonical_metric_key(key)
            if new_key != key:
                stats["renamed_values"] += 1
            if new_key in output:
                stats["collisions_kept_existing"] += 1
                continue
            output[new_key] = value
    stats["logged_rows"] = sum(1 for row in rows_by_step.values() if row)
    return rows_by_step, stats


def clone_run(
    source: Any,
    *,
    entity: str,
    project: str,
    name_suffix: str,
    history_samples: int,
    dry_run: bool,
) -> dict[str, Any]:
    rows_by_step, stats = transformed_history_rows(source, history_samples=history_samples)
    clone_name = f"{source.name}{name_suffix}"
    config = dict(source.config)
    config.update(
        {
            "metric_name_clone": True,
            "metric_name_clone_source_run_id": source.id,
            "metric_name_clone_source_name": source.name,
            "metric_name_clone_renames": dict(METRIC_RENAMES),
        }
    )
    tags = sorted(set(source.tags or []) | {"metric-name-clone", "canonical-metrics"})
    result: dict[str, Any] = {
        "source_id": source.id,
        "source_name": source.name,
        "clone_name": clone_name,
        **stats,
    }
    if dry_run:
        result["clone_id"] = None
        return result

    clone = wandb.init(
        entity=entity,
        project=project,
        name=clone_name,
        config=config,
        tags=tags,
        notes=f"Metric-name clone of {source.url}; legacy common keys canonicalized.",
        reinit=True,
    )
    assert clone is not None
    try:
        clone.define_metric("*", step_metric="_step")
        for step in sorted(rows_by_step):
            row = rows_by_step[step]
            if not row:
                continue
            clone.log({"_step": step, **row}, step=step)
        clone.summary["metric_name_clone_source_run_id"] = source.id
        clone.summary["metric_name_clone_source_url"] = source.url
        result["clone_id"] = clone.id
        result["clone_url"] = clone.url
    finally:
        clone.finish()
    return result


def iter_source_runs(api: Any, *, entity: str, project: str, run_ids: Iterable[str]) -> Iterable[Any]:
    for run_id in run_ids:
        yield api.run(f"{entity}/{project}/{run_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clone W&B runs while canonicalizing legacy metric names.")
    parser.add_argument("run_ids", nargs="+", help="Source W&B run ids to clone.")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name-suffix", default="-new-metrics")
    parser.add_argument("--history-samples", type=int, default=1_000_000)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    api = wandb.Api(timeout=120)
    for source in iter_source_runs(api, entity=args.entity, project=args.project, run_ids=args.run_ids):
        result = clone_run(
            source,
            entity=args.entity,
            project=args.project,
            name_suffix=args.name_suffix,
            history_samples=args.history_samples,
            dry_run=args.dry_run,
        )
        print(result, flush=True)


if __name__ == "__main__":
    main()
