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
from lfrm.training.brc import build_brc_act_train_step_runner
from lfrm.training.factory import create_model, create_optimizer


def _load_config(config_path: str, *, steps: int, evidence_steps: int | None, halt_loss_weight: float | None):
    parser = build_parser()
    parser.set_defaults(**load_toml_config(config_path))
    args = parser.parse_args(["--config", config_path])
    if args.dataset_path is None:
        raise ValueError("Config must provide [data].dataset_path")
    dataset = load_dataset(dataset_path=args.dataset_path)
    config = build_config(
        args,
        vocab_size=dataset.spec.vocab_size,
        num_puzzle_identifiers=dataset.spec.num_puzzle_identifiers,
        seq_len=dataset.spec.seq_len,
    )
    if config.model.model_type != "brc" or config.task.type != "sudoku":
        raise ValueError("This debug script expects a sudoku BRC config")
    brc = config.model.brc_config
    if evidence_steps is not None:
        brc = replace(brc, evidence_steps=evidence_steps)
    model = replace(config.model, brc=brc)
    train = replace(
        config.train,
        batch_size=1,
        optimizer_updates=steps,
        halt_loss_weight=config.train.halt_loss_weight if halt_loss_weight is None else halt_loss_weight,
    )
    runtime = replace(config.runtime, data_parallel_devices=1, prefetch_depth=1, prefetch_workers=1)
    return replace(config, model=model, train=train, runtime=runtime), dataset


def _make_single_batch(dataset, *, split: str, index: int) -> dict[str, jax.Array]:
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
    if index < 0 or index >= inputs.shape[0]:
        raise ValueError(f"--index={index} out of range for {split} split with {inputs.shape[0]} examples")
    batch = {
        "inputs": np.asarray(inputs[index : index + 1], dtype=np.int32),
        "labels": np.asarray(labels[index : index + 1], dtype=np.int32),
        "puzzle_identifiers": np.asarray(puzzle_identifiers[index : index + 1], dtype=np.int32),
        "example_mask": np.ones((1,), dtype=np.float32),
    }
    return jax.device_put(batch)


def _evidence_probs_np(evidence: np.ndarray) -> np.ndarray:
    alpha = np.maximum(evidence, 1e-12)
    return alpha / np.maximum(np.sum(alpha, axis=-1, keepdims=True), 1e-12)


def _probe(
    batch_host: dict[str, np.ndarray],
    carry_host: dict[str, np.ndarray],
    prev_pred: np.ndarray | None,
    prev_evidence: np.ndarray | None,
):
    inputs = batch_host["inputs"][0]
    labels = batch_host["labels"][0]
    evidence = carry_host["evidence"][0]
    pred = np.argmax(evidence, axis=-1).astype(np.int32)
    loss_mask = labels != 0
    context_mask = (inputs > 1) & loss_mask
    query_mask = (inputs <= 1) & loss_mask
    context_acc = float(np.mean(pred[context_mask] == labels[context_mask])) if np.any(context_mask) else 0.0
    query_acc = float(np.mean(pred[query_mask] == labels[query_mask])) if np.any(query_mask) else 0.0
    exact = bool(np.all(pred[loss_mask] == labels[loss_mask])) if np.any(loss_mask) else True
    probs = _evidence_probs_np(evidence)
    strength = np.sum(np.maximum(evidence, 1e-12), axis=-1, keepdims=True)
    uncertainty = np.minimum(evidence.shape[-1] / np.maximum(strength, 1e-12), 1.0)
    entropy = float(-np.mean(np.sum(probs[loss_mask] * np.log(np.maximum(probs[loss_mask], 1e-12)), axis=-1)))
    confidence = float(np.mean(np.max(probs[loss_mask], axis=-1)))
    query_changed = 0.0
    all_changed = 0.0
    if prev_pred is not None:
        all_changed = float(np.mean(pred[loss_mask] != prev_pred[loss_mask])) if np.any(loss_mask) else 0.0
        query_changed = float(np.mean(pred[query_mask] != prev_pred[query_mask])) if np.any(query_mask) else 0.0
    evidence_delta = 0.0
    if prev_evidence is not None:
        evidence_delta = float(np.sqrt(np.mean(np.square(evidence - prev_evidence))))
    return {
        "probe_context_accuracy": context_acc,
        "probe_query_accuracy": query_acc,
        "probe_exact": exact,
        "probe_entropy": entropy,
        "probe_confidence": confidence,
        "probe_evidence_strength": float(np.mean(strength[loss_mask])),
        "probe_evidence_uncertainty": float(np.mean(uncertainty[loss_mask])),
        "probe_changed": all_changed,
        "probe_query_changed": query_changed,
        "probe_evidence_delta_rms": evidence_delta,
    }, pred, evidence


def _metric(metrics: dict[str, object], name: str, default: float = 0.0) -> float:
    value = metrics.get(name, default)
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug BRC ACT on one fixed sudoku example.")
    parser.add_argument("--config", default="configs/sudoku_brc.toml")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--evidence-steps", type=int, default=None)
    parser.add_argument("--halt-loss-weight", type=float, default=None)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config, dataset = _load_config(
        args.config,
        steps=args.steps,
        evidence_steps=args.evidence_steps,
        halt_loss_weight=args.halt_loss_weight,
    )
    batch = _make_single_batch(dataset, split=args.split, index=args.index)
    batch_host = jax.device_get(batch)
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    train_step = build_brc_act_train_step_runner(config.train.halt_loss_weight)
    carry = model.initial_carry(batch)
    key = jax.random.key(args.seed)

    log_path = Path(args.log_path) if args.log_path else Path("debug_logs") / (
        f"single_brc_act_{args.split}_{args.index}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    prev_pred = None
    prev_evidence = None
    print(
        "step loss cur_q probe_q cur_ctx probe_ctx exact halt reset act entropy conf strength uncert dE qchg",
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
                probe, prev_pred, prev_evidence = _probe(
                    batch_host,
                    carry_host,
                    prev_pred,
                    prev_evidence,
                )
                row = {
                    "step": step,
                    "loss": _metric(host_metrics, "loss"),
                    "lm_loss": _metric(host_metrics, "lm_loss"),
                    "current_query_accuracy": _metric(host_metrics, "current_query_accuracy"),
                    "current_context_accuracy": _metric(host_metrics, "current_context_accuracy"),
                    "current_exact_accuracy": _metric(host_metrics, "current_exact_accuracy"),
                    "query_accuracy": _metric(host_metrics, "query_accuracy"),
                    "context_accuracy": _metric(host_metrics, "context_accuracy"),
                    "exact_accuracy": _metric(host_metrics, "exact_accuracy"),
                    "halted_rate": _metric(host_metrics, "halted_rate"),
                    "reset_rate": _metric(host_metrics, "reset_rate"),
                    "act_step": _metric(host_metrics, "act_step"),
                    "evidence_delta_mass": _metric(host_metrics, "evidence_delta_mass"),
                    "evidence_update_alpha": _metric(host_metrics, "evidence_update_alpha"),
                    **probe,
                }
                if "halt_loss" in host_metrics:
                    row["halt_loss"] = _metric(host_metrics, "halt_loss")
                log_file.write(json.dumps(row, sort_keys=True) + "\n")
                log_file.flush()
                print(
                    f"{step:5d} "
                    f"{row['loss']:.4f} "
                    f"{row['current_query_accuracy']:.3f} "
                    f"{row['probe_query_accuracy']:.3f} "
                    f"{row['current_context_accuracy']:.3f} "
                    f"{row['probe_context_accuracy']:.3f} "
                    f"{row['current_exact_accuracy']:.0f} "
                    f"{row['halted_rate']:.2f} "
                    f"{row['reset_rate']:.2f} "
                    f"{row['act_step']:.1f} "
                    f"{row['probe_entropy']:.2f} "
                    f"{row['probe_confidence']:.2f} "
                    f"{row['probe_evidence_strength']:.2f} "
                    f"{row['probe_evidence_uncertainty']:.2f} "
                    f"{row['evidence_delta_mass']:.2f} "
                    f"{row['evidence_update_alpha']:.2f} "
                    f"{row['probe_evidence_delta_rms']:.3e} "
                    f"{row['probe_query_changed']:.3f}",
                    flush=True,
                )
    print(f"[debug] wrote {log_path}", flush=True)


if __name__ == "__main__":
    main()
