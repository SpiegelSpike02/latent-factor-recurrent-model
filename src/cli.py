from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import tomllib
from typing import Any

import jax
import numpy as np

from config import (
    AttentionConfig,
    ClueConfig,
    ComputeConfig,
    DataConfig,
    EMAConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    RelationConfig,
    RuntimeConfig,
    TrainConfig,
    TransitionConfig,
    WandbConfig,
)
from data import dataset_overview, load_dataset, sample_batch
from training import (
    build_ema_update_runner,
    create_model,
    create_ema_model,
    create_optimizer,
    build_eval_step_runner,
    build_train_step_runner,
    save_checkpoint,
)


CONFIG_SECTIONS = ("data", "model", "optimizer", "train", "runtime", "wandb")
NESTED_SECTIONS = {
    "model": {
        "transition",
        "attention",
        "relation",
        "clues",
        "compute",
    },
    "train": {
        "ema",
    },
}


def load_toml_config(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open("rb") as f:
        loaded = tomllib.load(f)

    flat: dict[str, object] = {}
    for section in CONFIG_SECTIONS:
        section_values = loaded.get(section, {})
        if not isinstance(section_values, dict):
            raise ValueError(f"Section [{section}] in {config_path} must be a table")
        for key, value in section_values.items():
            normalized_key = key.replace("-", "_")
            if normalized_key in NESTED_SECTIONS.get(section, set()):
                if not isinstance(value, dict):
                    raise ValueError(f"Section [{section}.{key}] in {config_path} must be a table")
                for nested_key, nested_value in value.items():
                    flat[f"{normalized_key}_{nested_key.replace('-', '_')}"] = nested_value
                continue
            if section == "wandb":
                normalized_key = f"wandb_{normalized_key}"
            flat[normalized_key] = value
    return flat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train recurrent grid reasoning models.")
    parser.add_argument("--config", type=str, default=None, help="Optional TOML config file.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Offline grid dataset directory.")
    parser.add_argument("--seq-len", type=int, default=81)
    parser.add_argument("--grid-height", type=int, default=9)
    parser.add_argument("--grid-width", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--validity-loss-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ema-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--model-type",
        choices=("universal_transformer", "recurrent_transformer"),
        default="universal_transformer",
    )
    parser.add_argument("--communication-type", choices=("relation", "attention"), default="relation")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=6)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--transition-type", choices=("residual", "damped"), default="residual")
    parser.add_argument("--transition-hidden-dim", type=int, default=128)
    parser.add_argument("--attention-num-heads", type=int, default=4)
    parser.add_argument("--relation-include-global", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clues-use-type-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clues-fix-outputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clues-freeze-state", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compute-inner-steps", type=int, default=1)
    parser.add_argument("--compute-layers-per-step", type=int, default=1)
    parser.add_argument("--compute-grad-inner-steps", type=int, default=1)
    parser.add_argument("--compute-reinject-input", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--compute-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", type=str, default="ut-sudoku")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser


def build_config(args: argparse.Namespace, *, vocab_size: int, seq_len: int) -> ExperimentConfig:
    model = ModelConfig(
        vocab_size=vocab_size,
        model_type=args.model_type,
        communication_type=args.communication_type,
        seq_len=seq_len,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_steps=args.num_steps,
        dropout_rate=args.dropout_rate,
        transition=TransitionConfig(
            type=args.transition_type,
            hidden_dim=args.transition_hidden_dim,
        ),
        attention=AttentionConfig(
            num_heads=args.attention_num_heads,
        ),
        relation=RelationConfig(
            include_global=args.relation_include_global,
        ),
        clues=ClueConfig(
            use_type_embedding=args.clues_use_type_embedding,
            fix_outputs=args.clues_fix_outputs,
            freeze_state=args.clues_freeze_state,
        ),
        compute=ComputeConfig(
            inner_steps=args.compute_inner_steps,
            layers_per_step=args.compute_layers_per_step,
            grad_inner_steps=args.compute_grad_inner_steps,
            reinject_input=args.compute_reinject_input,
        ),
    )
    optimizer = OptimizerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm,
    )
    train = TrainConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        validity_loss_weight=args.validity_loss_weight,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        ema=EMAConfig(
            enabled=args.ema_enabled,
            decay=args.ema_decay,
        ),
    )
    data = DataConfig(
        dataset_path=args.dataset_path,
    )
    runtime = RuntimeConfig(compute_dtype=args.compute_dtype)
    wandb = WandbConfig(
        enabled=args.wandb_enabled,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
    )
    return ExperimentConfig(model=model, optimizer=optimizer, train=train, data=data, runtime=runtime, wandb=wandb)


def build_run_checkpoint_dir(config_path: str | None, checkpoint_root: str) -> Path:
    config_stem = Path(config_path).stem if config_path is not None else "run"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(checkpoint_root) / f"{config_stem}-{timestamp}"


def schedule_learning_rate(config: ExperimentConfig, step: int) -> float:
    warmup_steps = config.optimizer.warmup_steps
    peak = config.optimizer.learning_rate
    end = peak * 0.1
    decay_steps = max(config.train.max_steps, warmup_steps + 1)
    if step <= warmup_steps:
        return peak * step / max(warmup_steps, 1)
    progress = min(max((step - warmup_steps) / max(decay_steps - warmup_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(end + (peak - end) * cosine)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def init_wandb(config: ExperimentConfig, *, run_dir: Path):
    if not config.wandb.enabled or config.wandb.mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb logging is enabled but the 'wandb' package is not installed.") from exc

    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name,
        mode=config.wandb.mode,
        config=config_to_dict(config),
        dir=str(run_dir),
    )
    wandb.config.update({"checkpoint_dir": str(run_dir)}, allow_val_change=True)
    return run


def sample_device_batch(
    rng: np.random.Generator,
    dataset,
    *,
    config: ExperimentConfig,
    split: str,
    device: jax.Device,
) -> dict[str, jax.Array]:
    batch = sample_batch(
        rng,
        dataset,
        batch_size=config.train.batch_size,
        seq_len=config.model.seq_len,
        split=split,
    )
    return jax.device_put(batch, device=device)


def evaluate(eval_step_fn, model, dataset, *, config: ExperimentConfig, rng: np.random.Generator) -> dict[str, Any]:
    device = jax.devices()[0]
    reduced: dict[str, Any] | None = None
    for _ in range(config.train.eval_batches):
        batch = sample_device_batch(
            rng,
            dataset,
            config=config,
            split="eval",
            device=device,
        )
        metrics = eval_step_fn(model, batch)
        if reduced is None:
            reduced = {
                key: np.zeros(np.asarray(value).shape, dtype=np.float64)
                for key, value in metrics.items()
            }
        for key, value in metrics.items():
            reduced[key] += np.asarray(jax.device_get(value), dtype=np.float64)
    if reduced is None:
        raise ValueError("eval_batches must be at least 1")
    scale = 1.0 / config.train.eval_batches
    averaged = {key: value * scale for key, value in reduced.items()}
    return {
        key: float(value) if np.ndim(value) == 0 else value.astype(float).tolist()
        for key, value in averaged.items()
    }


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


def main() -> None:
    parser = build_parser()
    pre_args, _ = parser.parse_known_args()
    if pre_args.config is not None:
        parser.set_defaults(**load_toml_config(pre_args.config))
    args = parser.parse_args()
    if args.dataset_path is None:
        raise ValueError("--dataset-path is required")
    dataset = load_dataset(dataset_path=args.dataset_path)
    config = build_config(
        args,
        vocab_size=dataset.spec.vocab_size,
        seq_len=dataset.spec.seq_len,
    )

    checkpoint_dir = build_run_checkpoint_dir(args.config, config.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(config, run_dir=checkpoint_dir)

    model = create_model(config)
    optimizer = create_optimizer(model, config)
    train_step_fn = build_train_step_runner(config.train.validity_loss_weight)
    eval_step_fn = build_eval_step_runner(config.train.validity_loss_weight)
    ema_model = create_ema_model(model, config) if config.train.use_ema else None
    ema_update_fn = build_ema_update_runner(config.train.ema_decay) if config.train.use_ema else None

    train_rng = np.random.default_rng(config.train.seed)
    eval_rng = np.random.default_rng(config.train.seed + 1)
    train_key = jax.random.key(config.train.seed)

    device = jax.devices()[0]
    overview = dataset_overview(dataset)
    print(
        "device=", device,
        "dataset_kind=", overview["kind"],
        "task_type=", overview["task_type"],
        "vocab_size=", overview["vocab_size"],
        "train_examples=", overview["train_examples"],
        "eval_examples=", overview["eval_examples"],
        "seq_len=", config.model.seq_len,
        "grid_height=", config.model.grid_height,
        "grid_width=", config.model.grid_width,
        "checkpoint_dir=", checkpoint_dir,
    )

    current_batch = sample_device_batch(
        train_rng,
        dataset,
        config=config,
        split="train",
        device=device,
    )

    for step in range(1, config.train.max_steps + 1):
        train_key, step_key = jax.random.split(train_key)
        metrics = train_step_fn(
            model,
            optimizer,
            current_batch,
            step_key,
        )
        if ema_model is not None and ema_update_fn is not None:
            ema_update_fn(ema_model, model)

        next_batch = sample_device_batch(
            train_rng,
            dataset,
            config=config,
            split="train",
            device=device,
        )
        current_batch = next_batch

        if step % config.train.log_every == 0 or step == 1:
            train_log = {
                "train/loss": float(metrics["loss"]),
                "train/blank_ce_loss": float(metrics["blank_ce_loss"]),
                "train/validity_loss": float(metrics["validity_loss"]),
                "train/blank_cell_accuracy": float(metrics["blank_cell_accuracy"]),
                "train/solved_rate": float(metrics["solved_rate"]),
                "train/learning_rate": schedule_learning_rate(config, step),
            }
            if wandb_run is not None:
                wandb_run.log(train_log, step=step)
            print(
                f"step={step} "
                f"loss={float(metrics['loss']):.4f} "
                f"ce={float(metrics['blank_ce_loss']):.4f} "
                f"valid={float(metrics['validity_loss']):.4f} "
                f"blank_acc={float(metrics['blank_cell_accuracy']):.4f} "
                f"solved={float(metrics['solved_rate']):.4f}"
            )

        if step % config.train.eval_every == 0 or step == config.train.max_steps:
            eval_model = ema_model if ema_model is not None else model
            eval_metrics = evaluate(
                eval_step_fn,
                eval_model,
                dataset,
                config=config,
                rng=eval_rng,
            )
            if wandb_run is not None:
                eval_log = {
                    "eval/loss": eval_metrics["loss"],
                    "eval/blank_ce_loss": eval_metrics["blank_ce_loss"],
                    "eval/validity_loss": eval_metrics["validity_loss"],
                    "eval/blank_cell_accuracy": eval_metrics["blank_cell_accuracy"],
                    "eval/solved_rate": eval_metrics["solved_rate"],
                }
                eval_log.update(flatten_step_metrics("eval/loss_by_step", eval_metrics["per_step_loss"]))
                eval_log.update(
                    flatten_step_metrics(
                        "eval/validity_loss_by_step",
                        eval_metrics["per_step_validity_loss"],
                    )
                )
                if "per_step_rho" in eval_metrics:
                    eval_log.update(flatten_step_metrics("eval/rho_by_step", eval_metrics["per_step_rho"]))
                if "per_step_alpha" in eval_metrics:
                    eval_log.update(flatten_step_metrics("eval/alpha_by_step", eval_metrics["per_step_alpha"]))
                eval_log.update(
                    flatten_step_metrics(
                        "eval/hidden_delta_by_step",
                        eval_metrics["per_step_hidden_delta"],
                    )
                )
                wandb_run.log(
                    eval_log,
                    step=step,
                )
            print(
                f"[eval{'/ema' if ema_model is not None else ''}] step={step} "
                f"loss={eval_metrics['loss']:.4f} "
                f"ce={eval_metrics['blank_ce_loss']:.4f} "
                f"valid={eval_metrics['validity_loss']:.4f} "
                f"blank_acc={eval_metrics['blank_cell_accuracy']:.4f} "
                f"solved={eval_metrics['solved_rate']:.4f}"
            )
            step_summaries = [
                format_step_summary("loss", eval_metrics["per_step_loss"]),
                format_step_summary("valid", eval_metrics["per_step_validity_loss"]),
            ]
            if "per_step_alpha" in eval_metrics:
                step_summaries.append(format_step_summary("alpha", eval_metrics["per_step_alpha"]))
            if "per_step_rho" in eval_metrics:
                step_summaries.append(format_step_summary("rho", eval_metrics["per_step_rho"]))
            step_summaries.append(format_step_summary("delta", eval_metrics["per_step_hidden_delta"]))
            print("  " + " ".join(step_summaries))
            save_checkpoint(str(checkpoint_dir), model, optimizer, step, ema_model=ema_model)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
