from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import wandb


DEFAULT_ENTITY = "gjia0011-"
DEFAULT_PROJECT = "latent-factor-recurrent-model"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "minor-thesis-guixuan-2026" / "data" / "wandb_runs.json"
RUN_PREFIXES = ("sudoku-bdr", "maze-bdr", "sudoku-brc", "maze-brc", "arc")


SUMMARY_KEYS = (
    "_step",
    "_runtime",
    "train/loss",
    "train/token_loss",
    "train/query_accuracy",
    "train/query_target_probability",
    "train/exact_accuracy",
    "train/conflicts",
    "train/context_accuracy",
    "train/final_target_probability",
    "train/energy_grad_rms",
    "train/logit_step_rms",
    "train/distribution_tv_delta",
    "train/wrong_attractor_rank_loss",
    "train/wrong_attractor_energy_gap",
    "train/corrupted_recovery_loss",
    "eval/accuracy",
    "eval/query_accuracy",
    "eval/exact_accuracy",
    "eval/conflicts",
    "eval/ema/accuracy",
    "eval/ema/query_accuracy",
    "eval/ema/exact_accuracy",
    "eval/ema/conflicts",
)

HISTORY_KEYS = (
    "_step",
    "train/loss",
    "train/token_loss",
    "train/query_accuracy",
    "train/query_target_probability",
    "train/exact_accuracy",
    "train/conflicts",
    "train/context_accuracy",
    "train/final_target_probability",
    "train/energy_grad_rms",
    "train/logit_step_rms",
    "train/distribution_tv_delta",
    "train/wrong_attractor_rank_loss",
    "train/wrong_attractor_energy_gap",
    "train/corrupted_recovery_loss",
)


def scalar(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def config_get(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def config_get_any(config: dict[str, Any], *dotted_paths: str) -> Any:
    for dotted in dotted_paths:
        value = config_get(config, dotted)
        if value is not None:
            return value
    return None


def compact_history_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_rows: list[dict[str, Any]] = []
    for row in rows:
        compact_rows.append({key: scalar(row.get(key)) for key in HISTORY_KEYS if row.get(key) is not None})
    return compact_rows


def export_record(run: Any, *, history_samples: int) -> dict[str, Any]:
    config = dict(run.config)
    summary = {key: scalar(run.summary.get(key)) for key in SUMMARY_KEYS if key in run.summary}
    metadata = dict(run.metadata or {})
    history_rows = run.history(keys=list(HISTORY_KEYS), samples=history_samples, pandas=False)
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "url": run.url,
        "git_commit": metadata.get("git", {}).get("commit") or metadata.get("git_commit"),
        "config_summary": {
            "task": config_get(config, "task.type"),
            "model_type": config_get(config, "model.model_type"),
            "d_model": config_get(config, "model.d_model"),
            "loss_type": config_get(config, "model.loss_type"),
            "train_mode": config_get(config, "train.train_mode"),
            "batch_size": config_get(config, "train.batch_size"),
            "learning_rate": config_get(config, "optimizer.learning_rate"),
            "bdr_commit_steps": config_get_any(config, "model.bdr.commit_steps", "model.brc.commit_steps"),
            "bdr_refine_steps": config_get_any(config, "model.bdr.refine_steps", "model.brc.refine_steps"),
            "bdr_block_depth": config_get_any(config, "model.bdr.block_depth", "model.brc.block_depth"),
            "bdr_update_rule": config_get_any(config, "model.bdr.update_rule", "model.brc.update_rule"),
            "bdr_update_step_size": config_get_any(config, "model.bdr.update_step_size", "model.brc.update_step_size"),
            "bdr_wrong_attractor_rank_weight": config_get_any(
                config, "model.bdr.wrong_attractor_rank_weight", "model.brc.wrong_attractor_rank_weight"
            ),
            "bdr_wrong_attractor_direction_weight": config_get_any(
                config,
                "model.bdr.wrong_attractor_direction_weight",
                "model.brc.wrong_attractor_direction_weight",
            ),
            "bdr_corrupted_recovery_weight": config_get_any(
                config, "model.bdr.corrupted_recovery_weight", "model.brc.corrupted_recovery_weight"
            ),
        },
        "summary": summary,
        "history": compact_history_rows(history_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export compact W&B run metadata and sampled history.")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--history-samples", type=int, default=int(os.environ.get("WANDB_HISTORY_SAMPLES", "500")))
    parser.add_argument("--runs-per-page", type=int, default=int(os.environ.get("WANDB_RUNS_PER_PAGE", "20")))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    api = wandb.Api(timeout=120)
    count = 0
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        for run in api.runs(f"{args.entity}/{args.project}", order="-created_at", per_page=args.runs_per_page):
            if not run.name.startswith(RUN_PREFIXES):
                continue
            if count:
                handle.write(",\n")
            json.dump(export_record(run, history_samples=args.history_samples), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            count += 1
            print(f"exported {count}: {run.id} {run.name}", flush=True)
        handle.write("]\n")
    tmp.replace(out)
    print(f"wrote {out} with {count} runs")


if __name__ == "__main__":
    main()
