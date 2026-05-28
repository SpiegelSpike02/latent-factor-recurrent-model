from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from lfrm.cli import build_config, build_parser, load_toml_config
from lfrm.datasets import load_dataset
from lfrm.training.bdr import build_bdr_step_carry_train_step_runner
from lfrm.training.factory import create_model, create_optimizer


def _load_config(config_path: str, *, steps: int, commit_steps: int | None, batch_size: int):
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
    if config.model.model_type != "bdr" or config.task.type != "sudoku":
        raise ValueError("This debug script expects a sudoku BDR config")
    bdr = config.model.bdr_config
    if commit_steps is not None:
        bdr = replace(bdr, commit_steps=commit_steps)
    model = replace(config.model, bdr=bdr)
    train = replace(
        config.train,
        batch_size=batch_size,
        optimizer_updates=steps,
    )
    runtime = replace(config.runtime, data_parallel_devices=1, prefetch_depth=1, prefetch_workers=1)
    return replace(config, model=model, train=train, runtime=runtime), dataset


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
    if index < 0 or index >= inputs.shape[0]:
        raise ValueError(f"--index={index} out of range for {split} split with {inputs.shape[0]} examples")
    end = index + batch_size
    if end > inputs.shape[0]:
        raise ValueError(f"--index + --batch-size exceeds {split} split size {inputs.shape[0]}")
    batch = {
        "inputs": np.asarray(inputs[index:end], dtype=np.int32),
        "labels": np.asarray(labels[index:end], dtype=np.int32),
        "puzzle_identifiers": np.asarray(puzzle_identifiers[index:end], dtype=np.int32),
        "example_mask": np.ones((batch_size,), dtype=np.float32),
    }
    return jax.device_put(batch)


def _probe(
    batch_host: dict[str, np.ndarray],
    carry_host: dict[str, np.ndarray],
    prev_pred: np.ndarray | None,
    prev_q: np.ndarray | None,
):
    inputs = batch_host["inputs"]
    labels = batch_host["labels"]
    z_logits = carry_host["z"]
    z_shifted = z_logits - np.max(z_logits, axis=-1, keepdims=True)
    q_view = np.exp(z_shifted)
    q_view = q_view / np.maximum(np.sum(q_view, axis=-1, keepdims=True), 1e-12)
    pred = np.argmax(q_view, axis=-1).astype(np.int32)
    loss_mask = np.ones_like(labels, dtype=bool)
    context_mask = (inputs > 0) & loss_mask
    query_mask = (inputs == 0) & loss_mask
    context_acc = float(np.mean(pred[context_mask] == labels[context_mask])) if np.any(context_mask) else 0.0
    query_acc = float(np.mean(pred[query_mask] == labels[query_mask])) if np.any(query_mask) else 0.0
    exact_per_example = np.all((pred == labels) | ~loss_mask, axis=-1)
    exact = float(np.mean(exact_per_example)) if exact_per_example.size else 1.0
    entropy = float(-np.mean(np.sum(q_view[loss_mask] * np.log(np.maximum(q_view[loss_mask], 1e-12)), axis=-1)))
    confidence = float(np.mean(np.max(q_view[loss_mask], axis=-1)))
    query_changed = 0.0
    all_changed = 0.0
    if prev_pred is not None:
        all_changed = float(np.mean(pred[loss_mask] != prev_pred[loss_mask])) if np.any(loss_mask) else 0.0
        query_changed = float(np.mean(pred[query_mask] != prev_pred[query_mask])) if np.any(query_mask) else 0.0
    distribution_tv_delta = 0.0
    if prev_q is not None:
        distribution_tv_delta = float(np.mean(0.5 * np.sum(np.abs(q_view - prev_q), axis=-1)))
    return {
        "probe_context_accuracy": context_acc,
        "probe_query_accuracy": query_acc,
        "probe_exact_accuracy": exact,
        "probe_exact_count": float(np.sum(exact_per_example)),
        "probe_entropy": entropy,
        "probe_q_top1_probability": confidence,
        "probe_changed": all_changed,
        "probe_query_changed": query_changed,
        "probe_distribution_tv_delta": distribution_tv_delta,
    }, pred, q_view


def _metric(metrics: dict[str, object], name: str, default: float = 0.0) -> float:
    value = metrics.get(name, default)
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug BDR step-carry on a fixed sudoku mini-batch.")
    parser.add_argument("--config", default="configs/sudoku_bdr.toml")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--commit-steps", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config, dataset = _load_config(
        args.config,
        steps=args.steps,
        commit_steps=args.commit_steps,
        batch_size=args.batch_size,
    )
    batch = _make_batch(dataset, split=args.split, index=args.index, batch_size=args.batch_size)
    batch_host = jax.device_get(batch)
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    train_step = build_bdr_step_carry_train_step_runner()
    carry = model.initial_carry(batch)
    key = jax.random.key(args.seed)

    log_path = Path(args.log_path) if args.log_path else Path("debug_logs") / (
        f"single_bdr_step_carry_{args.split}_{args.index}_b{args.batch_size}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    prev_pred = None
    prev_q = None
    print(
        "step loss cur_q probe_q cur_ctx probe_ctx exact n max early stable e_dq flip margin cstr reset carry entropy conf eta update dz dQ energy qchg",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, args.steps + 1):
            key, step_key = jax.random.split(key)
            metrics, carry = train_step(
                model,
                optimizer,
                carry,
                batch,
                step_key,
                jnp.asarray(step - 1, dtype=jnp.int32),
            )
            should_print = step == 1 or step % args.print_every == 0 or step == args.steps
            if should_print:
                host_metrics = jax.device_get(metrics)
                carry_host = jax.device_get(carry)
                probe, prev_pred, prev_q = _probe(
                    batch_host,
                    carry_host,
                    prev_pred,
                    prev_q,
                )
                row = {
                    "step": step,
                    "loss": _metric(host_metrics, "loss"),
                    "token_loss": _metric(host_metrics, "token_loss"),
                    "query_accuracy": _metric(host_metrics, "query_accuracy"),
                    "context_accuracy": _metric(host_metrics, "context_accuracy"),
                    "exact_accuracy": _metric(host_metrics, "exact_accuracy"),
                    "max_step_reset_rate": _metric(host_metrics, "max_step_reset_rate"),
                    "early_stop_rate": _metric(host_metrics, "early_stop_rate"),
                    "stability_rate": _metric(host_metrics, "stability_rate"),
                    "stable_steps": _metric(host_metrics, "stable_steps"),
                    "early_stop_update_rms": _metric(host_metrics, "early_stop_update_rms"),
                    "early_stop_distribution_delta": _metric(host_metrics, "early_stop_distribution_delta"),
                    "early_stop_flip_rate": _metric(host_metrics, "early_stop_flip_rate"),
                    "early_stop_margin_min": _metric(host_metrics, "early_stop_margin_min"),
                    "early_stop_constraint_rate": _metric(host_metrics, "early_stop_constraint_rate"),
                    "reset_rate": _metric(host_metrics, "reset_rate"),
                    "carry_step": _metric(host_metrics, "carry_step"),
                    "update_step_size": _metric(host_metrics, "update_step_size"),
                    "update_rms": _metric(host_metrics, "update_rms"),
                    "logit_step_rms": _metric(host_metrics, "logit_step_rms"),
                    "path_energy": _metric(host_metrics, "path_energy"),
                    "fixed_point_update_loss": _metric(host_metrics, "fixed_point_update_loss"),
                    "wrong_attractor_rank_loss": _metric(host_metrics, "wrong_attractor_rank_loss"),
                    "wrong_attractor_direction_loss": _metric(host_metrics, "wrong_attractor_direction_loss"),
                    "wrong_attractor_nonzero_loss": _metric(host_metrics, "wrong_attractor_nonzero_loss"),
                    "wrong_attractor_active_rate": _metric(host_metrics, "wrong_attractor_active_rate"),
                    "wrong_attractor_direction_cosine": _metric(host_metrics, "wrong_attractor_direction_cosine"),
                    "wrong_attractor_energy_gap": _metric(host_metrics, "wrong_attractor_energy_gap"),
                    "corrupted_recovery_loss": _metric(host_metrics, "corrupted_recovery_loss"),
                    "corrupted_recovery_rank_loss": _metric(host_metrics, "corrupted_recovery_rank_loss"),
                    "corrupted_recovery_direction_cosine": _metric(host_metrics, "corrupted_recovery_direction_cosine"),
                    "corrupted_recovery_energy_gap": _metric(host_metrics, "corrupted_recovery_energy_gap"),
                    **probe,
                }
                log_file.write(json.dumps(row, sort_keys=True) + "\n")
                log_file.flush()
                print(
                    f"{step:5d} "
                    f"{row['loss']:.4f} "
                    f"{row['query_accuracy']:.3f} "
                    f"{row['probe_query_accuracy']:.3f} "
                    f"{row['context_accuracy']:.3f} "
                    f"{row['probe_context_accuracy']:.3f} "
                    f"{row['exact_accuracy']:.3f} "
                    f"{row['probe_exact_count']:.0f} "
                    f"{row['max_step_reset_rate']:.2f} "
                    f"{row['early_stop_rate']:.2f} "
                    f"{row['stability_rate']:.2f} "
                    f"{row['early_stop_update_rms']:.2e} "
                    f"{row['early_stop_distribution_delta']:.2e} "
                    f"{row['early_stop_flip_rate']:.2f} "
                    f"{row['early_stop_margin_min']:.2f} "
                    f"{row['early_stop_constraint_rate']:.2f} "
                    f"{row['reset_rate']:.2f} "
                    f"{row['carry_step']:.1f} "
                    f"{row['probe_entropy']:.2f} "
                    f"{row['probe_q_top1_probability']:.2f} "
                    f"{row['update_step_size']:.2f} "
                    f"{row['update_rms']:.2f} "
                    f"{row['logit_step_rms']:.2f} "
                    f"{row['probe_distribution_tv_delta']:.3e} "
                    f"{row['path_energy']:.3e} "
                    f"{row['probe_query_changed']:.3f}",
                    flush=True,
                )
    print(f"[debug] wrote {log_path}", flush=True)


if __name__ == "__main__":
    main()
