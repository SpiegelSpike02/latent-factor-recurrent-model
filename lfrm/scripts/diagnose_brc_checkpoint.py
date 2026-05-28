from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from lfrm.cli import build_config, build_parser, load_toml_config
from lfrm.datasets import load_dataset
from lfrm.training.checkpointing import load_checkpoint
from lfrm.training.factory import create_model, create_optimizer


def _load_config(config_path: str, *, batch_size: int, commit_steps: int | None):
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
        raise ValueError("This diagnostic script expects model_type='brc'")
    brc = config.model.brc_config
    if commit_steps is not None:
        brc = replace(brc, commit_steps=commit_steps)
    config = replace(
        config,
        model=replace(config.model, brc=brc),
        train=replace(config.train, batch_size=batch_size),
        runtime=replace(config.runtime, data_parallel_devices=1, prefetch_depth=1, prefetch_workers=1),
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
    if not 0 <= index < inputs.shape[0]:
        raise ValueError(f"--index={index} out of range for {split} split with {inputs.shape[0]} examples")
    end = index + batch_size
    if end > inputs.shape[0]:
        raise ValueError(f"--index + --batch-size exceeds {split} split size {inputs.shape[0]}")
    return jax.device_put(
        {
            "inputs": np.asarray(inputs[index:end], dtype=np.int32),
            "labels": np.asarray(labels[index:end], dtype=np.int32),
            "puzzle_identifiers": np.asarray(puzzle_identifiers[index:end], dtype=np.int32),
            "example_mask": np.ones((batch_size,), dtype=np.float32),
        }
    )


def _masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask_f = mask.astype(jnp.float32)
    return jnp.sum(values.astype(jnp.float32) * mask_f) / jnp.maximum(jnp.sum(mask_f), 1.0)


def _masked_rms(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask_f = mask.astype(jnp.float32)[..., None]
    return jnp.sqrt(
        jnp.sum(jnp.square(values.astype(jnp.float32)) * mask_f)
        / jnp.maximum(jnp.sum(mask_f) * values.shape[-1], 1.0)
        + 1e-12
    )


def _direction_cosine(step: jax.Array, target_delta: jax.Array, mask: jax.Array) -> jax.Array:
    mask_f = mask.astype(jnp.float32)[..., None]
    dot = jnp.sum(step.astype(jnp.float32) * target_delta.astype(jnp.float32) * mask_f)
    step_norm = jnp.sqrt(jnp.sum(jnp.square(step.astype(jnp.float32)) * mask_f) + 1e-12)
    target_norm = jnp.sqrt(jnp.sum(jnp.square(target_delta.astype(jnp.float32)) * mask_f) + 1e-12)
    return dot / (step_norm * target_norm + 1e-12)


def _sudoku_conflicts(pred: jax.Array, inputs: jax.Array, vocab_size: int) -> jax.Array:
    del inputs
    grid = pred.reshape((pred.shape[0], 9, 9))
    row_counts = jnp.sum(jax.nn.one_hot(grid, vocab_size), axis=2)
    col_counts = jnp.sum(jax.nn.one_hot(grid, vocab_size), axis=1)
    boxes = grid.reshape((pred.shape[0], 3, 3, 3, 3)).transpose((0, 1, 3, 2, 4)).reshape((pred.shape[0], 9, 9))
    box_counts = jnp.sum(jax.nn.one_hot(boxes, vocab_size), axis=2)
    row_conflicts = jnp.sum(jnp.maximum(row_counts - 1, 0), axis=(1, 2))
    col_conflicts = jnp.sum(jnp.maximum(col_counts - 1, 0), axis=(1, 2))
    box_conflicts = jnp.sum(jnp.maximum(box_counts - 1, 0), axis=(1, 2))
    return row_conflicts + col_conflicts + box_conflicts


def build_diagnostic_runner(model):
    @nnx.jit
    def run(batch: dict[str, jax.Array], key: jax.Array) -> dict[str, Any]:
        del key
        tokens = batch["inputs"]
        targets = batch["labels"]
        example_mask = batch.get("example_mask", jnp.ones((tokens.shape[0],), dtype=jnp.float32)).astype(jnp.float32)
        cell_mask = jnp.ones_like(targets, dtype=bool)
        context_mask = model.context_mask(tokens)
        query_mask = (~context_mask) & cell_mask
        example_cell_mask = example_mask[:, None] > 0
        query_mask = query_mask & example_cell_mask
        supervised_mask = cell_mask & example_cell_mask
        base_embeddings, _context = model.context_memory(tokens)
        target_logits = model.target_z_logits(targets)

        def eval_state(name: str, old_z: jax.Array, new_z: jax.Array, diagnostics: dict[str, jax.Array]) -> dict[str, jax.Array]:
            old_logits = model._center_logits(old_z)
            new_logits = model._center_logits(new_z)
            old_dist = model._distribution_from_logits(old_logits)
            new_dist = model._distribution_from_logits(new_logits)
            pred = jnp.argmax(new_dist, axis=-1)
            correct = (pred == targets) & supervised_mask
            query_correct = (pred == targets) & query_mask
            exact = jnp.all((pred == targets) | ~supervised_mask, axis=-1)
            top2 = jnp.sort(new_dist, axis=-1)[..., -2:]
            margin = top2[..., 1] - top2[..., 0]
            target_prob = jnp.take_along_axis(new_dist, targets[..., None], axis=-1)[..., 0]
            logit_step = new_logits - old_logits
            target_delta = model._center_logits(target_logits - old_logits)
            flip = jnp.argmax(old_dist, axis=-1) != pred
            if model.config.task_type == "sudoku":
                constraint_ok = model._sudoku_constraint_ok(pred, tokens)
                conflicts = _sudoku_conflicts(pred, tokens, model.q_vocab_size)
            else:
                constraint_ok = jnp.ones((tokens.shape[0],), dtype=bool)
                conflicts = jnp.zeros((tokens.shape[0],), dtype=jnp.float32)
            return {
                f"{name}/ce": _masked_mean(-jnp.log(jnp.maximum(target_prob, 1e-9)), supervised_mask),
                f"{name}/accuracy": _masked_mean(correct.astype(jnp.float32), supervised_mask),
                f"{name}/query_accuracy": _masked_mean(query_correct.astype(jnp.float32), query_mask),
                f"{name}/exact_accuracy": jnp.sum(exact.astype(jnp.float32) * example_mask) / jnp.maximum(jnp.sum(example_mask), 1.0),
                f"{name}/exact_count": jnp.sum(exact.astype(jnp.float32) * example_mask),
                f"{name}/target_probability": _masked_mean(target_prob, supervised_mask),
                f"{name}/q_top1_probability": _masked_mean(jnp.max(new_dist, axis=-1), supervised_mask),
                f"{name}/margin_min_query": jnp.min(jnp.where(query_mask, margin, jnp.inf)),
                f"{name}/flip_rate_query": _masked_mean(flip.astype(jnp.float32), query_mask),
                f"{name}/constraint_rate": jnp.sum(constraint_ok.astype(jnp.float32) * example_mask) / jnp.maximum(jnp.sum(example_mask), 1.0),
                f"{name}/conflicts": jnp.sum(conflicts.astype(jnp.float32) * example_mask) / jnp.maximum(jnp.sum(example_mask), 1.0),
                f"{name}/logit_step_rms_query": _masked_rms(logit_step, query_mask),
                f"{name}/direction_cosine_to_target": _direction_cosine(logit_step, target_delta, query_mask),
                f"{name}/energy_value": jnp.mean(diagnostics["energy_value"].astype(jnp.float32)),
                f"{name}/energy_grad_rms": jnp.mean(diagnostics["energy_grad_rms"].astype(jnp.float32)),
                f"{name}/update_rms": jnp.mean(diagnostics["update_rms"].astype(jnp.float32)),
                f"{name}/distribution_tv_delta": jnp.mean(diagnostics["distribution_tv_delta"].astype(jnp.float32)),
                f"{name}/path_energy": jnp.mean(diagnostics["path_energy"].astype(jnp.float32)),
            }

        def one_commit(
            z: jax.Array,
            hidden: jax.Array,
            step_index: jax.Array,
            *,
            stop_hidden: bool = True,
        ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
            return model._commit_step(
                tokens,
                z,
                hidden,
                base_embeddings,
                step_index,
                train=False,
                dropout_key=jax.random.key(0),
                stop_hidden_between_steps=stop_hidden,
            )

        def rollout(initial_z: jax.Array):
            hidden = model.initial_hidden_state(
                tokens,
                initial_z,
                base_embeddings,
                train=False,
                dropout_key=jax.random.key(0),
            )

            def scan_step(carry, step_index):
                z, h = carry
                next_z, h_next, diag = one_commit(z, h, step_index)
                return (next_z, h_next), diag

            final_carry, diagnostics = jax.lax.scan(
                scan_step,
                (initial_z, hidden),
                jnp.arange(model.total_steps, dtype=jnp.int32),
            )
            return final_carry, diagnostics

        uniform_z = model.initial_z(tokens)
        (plateau_z, plateau_hidden), rollout_diagnostics = rollout(uniform_z)
        last_rollout_diag = jax.tree.map(lambda x: x[-1], rollout_diagnostics)
        results = {}
        results.update(eval_state("uniform_rollout", uniform_z, plateau_z, last_rollout_diag))
        results["uniform_rollout/mean_energy_grad_rms"] = jnp.mean(rollout_diagnostics["energy_grad_rms"].astype(jnp.float32))
        results["uniform_rollout/mean_distribution_tv_delta"] = jnp.mean(rollout_diagnostics["distribution_tv_delta"].astype(jnp.float32))
        results["uniform_rollout/final_step_energy_grad_rms"] = jnp.mean(last_rollout_diag["energy_grad_rms"].astype(jnp.float32))

        final_step = jnp.asarray(model.total_steps - 1, dtype=jnp.int32)

        target_hidden = model.initial_hidden_state(tokens, target_logits, base_embeddings, train=False, dropout_key=jax.random.key(0))
        target_next, _target_hidden_next, target_diag = one_commit(target_logits, target_hidden, final_step)
        results.update(eval_state("target_fixed_point_probe", target_logits, target_next, target_diag))

        wrong_labels = jnp.mod(targets + 1, model.q_vocab_size)
        wrong_distribution = (
            (1.0 - float(model.brc.fixed_point_label_smoothing))
            * jax.nn.one_hot(wrong_labels, model.q_vocab_size, dtype=jnp.float32)
            + float(model.brc.fixed_point_label_smoothing) / float(model.q_vocab_size)
        )
        wrong_logits = model._center_logits(jnp.log(jnp.maximum(wrong_distribution, model.output_logit_eps)))
        wrong_hidden = model.initial_hidden_state(tokens, wrong_logits, base_embeddings, train=False, dropout_key=jax.random.key(0))
        wrong_next, _wrong_hidden_next, wrong_diag = one_commit(wrong_logits, wrong_hidden, final_step)
        results.update(eval_state("wrong_high_conf_probe", wrong_logits, wrong_next, wrong_diag))

        plateau_next, _plateau_hidden_next, plateau_diag = one_commit(
            plateau_z,
            plateau_hidden,
            final_step,
        )
        results.update(eval_state("plateau_carry_probe", plateau_z, plateau_next, plateau_diag))

        reset_hidden = model.initial_hidden_state(tokens, plateau_z, base_embeddings, train=False, dropout_key=jax.random.key(0))
        plateau_reset_h_next, _reset_h_hidden_next, reset_h_diag = one_commit(
            plateau_z,
            reset_hidden,
            final_step,
        )
        results.update(eval_state("plateau_reset_h_probe", plateau_z, plateau_reset_h_next, reset_h_diag))
        return results

    return run


def _hostify(results: dict[str, Any]) -> dict[str, float]:
    host = jax.device_get(results)
    return {key: float(np.asarray(value).reshape(-1)[0]) for key, value in sorted(host.items())}


def _print_table(results: dict[str, float]) -> None:
    scenarios = [
        "uniform_rollout",
        "plateau_carry_probe",
        "plateau_reset_h_probe",
        "plateau_reset_t_probe",
        "wrong_high_conf_probe",
        "target_fixed_point_probe",
    ]
    columns = [
        "ce",
        "query_accuracy",
        "exact_accuracy",
        "conflicts",
        "constraint_rate",
        "energy_grad_rms",
        "logit_step_rms_query",
        "distribution_tv_delta",
        "direction_cosine_to_target",
        "energy_value",
    ]
    print("\nscenario " + " ".join(f"{name:>23}" for name in columns))
    for scenario in scenarios:
        values = [results.get(f"{scenario}/{column}", float("nan")) for column in columns]
        print(f"{scenario:24s} " + " ".join(f"{value:23.6g}" for value in values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose BRC checkpoint attractors on a fixed batch.")
    parser.add_argument("--config", default="configs/sudoku_brc.toml")
    parser.add_argument("--checkpoint", required=True, help="Concrete step_N checkpoint path.")
    parser.add_argument("--split", choices=("train", "eval"), default="eval")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--commit-steps", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--use-ema", action="store_true", help="Load checkpoint EMA weights when present.")
    args = parser.parse_args()

    config, dataset = _load_config(args.config, batch_size=args.batch_size, commit_steps=args.commit_steps)
    batch = _make_batch(dataset, split=args.split, index=args.index, batch_size=args.batch_size)
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    ema_model = create_model(config) if args.use_ema else None
    restored_step = load_checkpoint(args.checkpoint, model, optimizer, ema_model=ema_model)
    if ema_model is not None:
        model = ema_model

    runner = build_diagnostic_runner(model)
    results = _hostify(runner(batch, jax.random.key(0)))
    results["checkpoint_step"] = float(restored_step)
    results["batch_size"] = float(args.batch_size)
    results["index"] = float(args.index)
    _print_table(results)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\n[diagnose] wrote {output}")


if __name__ == "__main__":
    main()
