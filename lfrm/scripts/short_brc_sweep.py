from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from lfrm.cli import build_config, build_parser, load_toml_config
from lfrm.datasets import load_dataset
from lfrm.training.brc import build_brc_step_carry_train_step_runner
from lfrm.training.factory import create_model, create_optimizer


KEY_METRICS = (
    "loss",
    "ce_loss",
    "accuracy",
    "query_accuracy",
    "context_accuracy",
    "exact_accuracy",
    "final_target_probability",
    "query_target_probability",
    "conflicts",
    "carry_step",
    "distribution_tv_delta",
    "path_energy",
    "update_rms",
    "energy_grad_rms",
    "logit_step_rms",
    "wrong_attractor_rank_loss",
    "wrong_attractor_direction_loss",
    "wrong_attractor_nonzero_loss",
    "wrong_attractor_direction_cosine",
    "wrong_attractor_energy_gap",
    "corrupted_recovery_loss",
    "corrupted_recovery_direction_cosine",
    "corrupted_recovery_energy_gap",
)


def _metric(metrics: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = metrics.get(name, default)
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0])


def _load_config(
    config_path: str,
    *,
    update_rule: str,
    update_step_size: float,
    steps: int,
    batch_size: int,
    commit_steps: int | None,
    seed: int,
):
    parser = build_parser()
    parser.set_defaults(**load_toml_config(config_path))
    args = parser.parse_args(["--config", config_path])
    if args.dataset_path is None:
        raise ValueError("Config must provide [data].dataset_path")
    dataset = load_dataset(dataset_path=args.dataset_path)
    config = build_config(
        args,
        vocab_size=dataset.spec.vocab_size,
        input_vocab_size=dataset.spec.input_vocab_size,
        num_puzzle_identifiers=dataset.spec.num_puzzle_identifiers,
        seq_len=dataset.spec.seq_len,
    )
    if config.model.model_type != "brc":
        raise ValueError("This sweep expects model_type='brc'")
    brc = replace(
        config.model.brc_config,
        update_rule=update_rule,
        update_step_size=update_step_size,
    )
    if commit_steps is not None:
        brc = replace(brc, commit_steps=commit_steps)
    config = replace(
        config,
        model=replace(config.model, brc=brc),
        train=replace(
            config.train,
            batch_size=batch_size,
            optimizer_updates=steps,
            seed=seed,
        ),
        runtime=replace(
            config.runtime,
            data_parallel_devices=1,
            prefetch_depth=1,
            prefetch_workers=1,
        ),
    )
    return config, dataset


def _make_batch(dataset, *, split: str, index: int, batch_size: int) -> dict[str, jax.Array]:
    if split == "train":
        inputs = dataset.train_inputs
        labels = dataset.train_labels
        puzzle_identifiers = dataset.train_puzzle_identifiers
    elif split == "eval":
        inputs = dataset.eval_inputs
        labels = dataset.eval_labels
        puzzle_identifiers = dataset.eval_puzzle_identifiers
    else:
        raise ValueError("--split must be train or eval")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if index < 0 or index + batch_size > inputs.shape[0]:
        raise ValueError(f"Requested [{index}, {index + batch_size}) outside {split} split of size {inputs.shape[0]}")
    return jax.device_put(
        {
            "inputs": np.asarray(inputs[index : index + batch_size], dtype=np.int32),
            "labels": np.asarray(labels[index : index + batch_size], dtype=np.int32),
            "puzzle_identifiers": np.asarray(puzzle_identifiers[index : index + batch_size], dtype=np.int32),
            "example_mask": np.ones((batch_size,), dtype=np.float32),
        }
    )


def _summarize(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {}
    best_query = max(rows, key=lambda row: row.get("query_accuracy", float("-inf")))
    best_prob = max(rows, key=lambda row: row.get("query_target_probability", float("-inf")))
    best_loss = min(rows, key=lambda row: row.get("loss", float("inf")))
    final = rows[-1]
    return {
        "final": final,
        "best_query_accuracy": best_query,
        "best_query_target_probability": best_prob,
        "best_loss": best_loss,
    }


def _run_trial(
    *,
    config_path: str,
    output_dir: Path,
    update_rule: str,
    update_step_size: float,
    steps: int,
    batch_size: int,
    commit_steps: int | None,
    split: str,
    index: int,
    seed: int,
    log_every: int,
) -> dict[str, Any]:
    config, dataset = _load_config(
        config_path,
        update_rule=update_rule,
        update_step_size=update_step_size,
        steps=steps,
        batch_size=batch_size,
        commit_steps=commit_steps,
        seed=seed,
    )
    batch = _make_batch(dataset, split=split, index=index, batch_size=batch_size)
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    train_step = build_brc_step_carry_train_step_runner()
    carry = model.initial_carry(batch)
    key = jax.random.key(seed)

    tag = f"{update_rule}_eta{update_step_size:g}_b{batch_size}_u{steps}_c{config.model.brc_config.commit_steps}_s{seed}"
    log_path = output_dir / f"{tag}.jsonl"
    rows: list[dict[str, float]] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, steps + 1):
            key, step_key = jax.random.split(key)
            metrics, carry = train_step(
                model,
                optimizer,
                carry,
                batch,
                step_key,
                jnp.asarray(step - 1, dtype=jnp.int32),
            )
            if step == 1 or step % log_every == 0 or step == steps:
                host_metrics = jax.device_get(metrics)
                row = {
                    "step": float(step),
                    "update_rule": update_rule,
                    "update_step_size": float(update_step_size),
                    "commit_steps": float(config.model.brc_config.commit_steps),
                }
                row.update({name: _metric(host_metrics, name) for name in KEY_METRICS})
                rows.append(row)
                log_file.write(json.dumps(row, sort_keys=True) + "\n")
                log_file.flush()
    summary = _summarize(rows)
    summary.update(
        {
            "update_rule": update_rule,
            "update_step_size": update_step_size,
            "log_path": str(log_path),
            "steps": steps,
            "batch_size": batch_size,
            "commit_steps": config.model.brc_config.commit_steps,
            "seed": seed,
        }
    )
    return summary


def _print_summary(summaries: list[dict[str, Any]]) -> None:
    print(
        "eta    best_qacc best_qprob best_loss final_qacc final_qprob final_loss conflicts log",
        flush=True,
    )
    for summary in summaries:
        best_q = summary["best_query_accuracy"]
        best_p = summary["best_query_target_probability"]
        best_l = summary["best_loss"]
        final = summary["final"]
        print(
            f"{summary['update_step_size']:<6g} "
            f"{best_q['query_accuracy']:.4f}    "
            f"{best_p['query_target_probability']:.4f}     "
            f"{best_l['loss']:.4f}    "
            f"{final['query_accuracy']:.4f}     "
            f"{final['query_target_probability']:.4f}      "
            f"{final['loss']:.4f}    "
            f"{final['conflicts']:.1f}     "
            f"{summary['log_path']}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a very short BRC sweep and summarize train metrics.")
    parser.add_argument("--config", default="configs/sudoku_brc.toml")
    parser.add_argument("--output-dir", default="logs/ablations/short_brc_sweep")
    parser.add_argument("--update-rule", choices=("energy", "velocity"), default="energy")
    parser.add_argument("--update-step-sizes", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--commit-steps", type=int, default=8)
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    started_at = time.strftime("%Y%m%d-%H%M%S")
    for update_step_size in args.update_step_sizes:
        print(f"[short-sweep] {args.update_rule} eta={update_step_size:g}", flush=True)
        summaries.append(
            _run_trial(
                config_path=args.config,
                output_dir=output_dir,
                update_rule=args.update_rule,
                update_step_size=update_step_size,
                steps=args.steps,
                batch_size=args.batch_size,
                commit_steps=args.commit_steps,
                split=args.split,
                index=args.index,
                seed=args.seed,
                log_every=args.log_every,
            )
        )
    summary_path = output_dir / f"summary_{args.update_rule}_{started_at}.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary(summaries)
    print(f"[short-sweep] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
