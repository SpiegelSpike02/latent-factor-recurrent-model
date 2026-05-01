from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import tomllib
from typing import Any

import jax
import numpy as np

from lfrm.config import (
    DataConfig,
    EMAConfig,
    ExperimentConfig,
    LFRMConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    TrainConfig,
    WandbConfig,
)
from lfrm.datasets import dataset_overview, load_dataset, sample_batch
from lfrm.training import (
    build_ema_update_runner,
    build_eval_step_runner,
    build_train_step_runner,
    create_ema_model,
    create_model,
    create_optimizer,
    load_checkpoint,
    save_checkpoint,
)


CONFIG_SECTIONS = ("data", "model", "optimizer", "train", "runtime", "wandb")
NESTED_SECTIONS = {
    "model": {"lfrm"},
    "train": {"ema"},
}
ALLOWED_SECTION_KEYS = {
    "data": {"dataset_path"},
    "model": {
        "model_type",
        "seq_len",
        "grid_height",
        "grid_width",
        "d_model",
        "num_steps",
        "dropout_rate",
        "lfrm",
    },
    "optimizer": {
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "grad_clip_norm",
        "flatten_optimizer",
    },
    "train": {
        "batch_size",
        "max_steps",
        "log_every",
        "eval_every",
        "eval_batches",
        "step_loss_weighting",
        "terminal_residual_weight",
        "energy_loss_weight",
        "energy_margin",
        "energy_corruptions",
        "slot_consistency_weight",
        "slot_usage_weight",
        "slot_diversity_weight",
        "seed",
        "checkpoint_dir",
        "ema",
    },
    "runtime": {"compute_dtype"},
    "wandb": {"enabled", "project", "entity", "name", "mode"},
}
ALLOWED_NESTED_KEYS = {
    "lfrm": {
        "belief_dim",
        "num_slots",
        "num_branches",
        "num_heads",
        "latent_processor_layers",
        "symbol_context_mode",
        "slot_readout_mode",
        "energy_symbol_pooling",
        "branch_diversity_schedule",
        "diversity_apply_steps",
        "belief_temperature",
        "belief_step_size",
        "belief_floor",
        "assignment_temperature",
        "energy_hidden_dim",
        "use_condition_type_embedding",
    },
    "ema": {"enabled", "decay"},
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
        allowed_keys = ALLOWED_SECTION_KEYS[section]
        for key, value in section_values.items():
            normalized_key = key.replace("-", "_")
            if normalized_key not in allowed_keys:
                raise ValueError(f"Unsupported [{section}] field in LFRM config: {key}")
            if normalized_key in NESTED_SECTIONS.get(section, set()):
                if not isinstance(value, dict):
                    raise ValueError(f"Section [{section}.{key}] in {config_path} must be a table")
                for nested_key, nested_value in value.items():
                    normalized_nested_key = nested_key.replace("-", "_")
                    if normalized_nested_key not in ALLOWED_NESTED_KEYS[normalized_key]:
                        raise ValueError(f"Unsupported [{section}.{key}] field in LFRM config: {nested_key}")
                    flat[f"{normalized_key}_{normalized_nested_key}"] = nested_value
                continue
            if section == "wandb":
                normalized_key = f"wandb_{normalized_key}"
            flat[normalized_key] = value

    if flat.get("model_type", "lfrm") != "lfrm":
        raise ValueError("Only model_type='lfrm' is supported")
    return flat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Latent Factor Recurrent Model.")
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
    parser.add_argument(
        "--step-loss-weighting",
        choices=("uniform", "linear", "final"),
        default="final",
        help="How to weight losses from recurrent reasoning steps.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--terminal-residual-weight", type=float, default=0.0)
    parser.add_argument("--energy-loss-weight", type=float, default=0.0)
    parser.add_argument("--energy-margin", type=float, default=1.0)
    parser.add_argument("--energy-corruptions", type=int, default=1)
    parser.add_argument("--slot-consistency-weight", type=float, default=0.0)
    parser.add_argument("--slot-usage-weight", type=float, default=0.0)
    parser.add_argument("--slot-diversity-weight", type=float, default=0.0)
    parser.add_argument("--ema-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--model-type", choices=("lfrm",), default="lfrm")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=6)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--lfrm-belief-dim", type=int, default=0)
    parser.add_argument("--lfrm-num-slots", type=int, default=64)
    parser.add_argument("--lfrm-num-branches", type=int, default=4)
    parser.add_argument("--lfrm-num-heads", type=int, default=4)
    parser.add_argument("--lfrm-latent-processor-layers", type=int, default=1)
    parser.add_argument("--lfrm-symbol-context-mode", choices=("cell_symbol_tokens",), default="cell_symbol_tokens")
    parser.add_argument("--lfrm-slot-readout-mode", choices=("cell_symbol_attention",), default="cell_symbol_attention")
    parser.add_argument("--lfrm-energy-symbol-pooling", choices=("deepsets",), default="deepsets")
    parser.add_argument("--lfrm-branch-diversity-schedule", choices=("early", "none"), default="early")
    parser.add_argument("--lfrm-diversity-apply-steps", type=int, nargs=2, default=(0, 8))
    parser.add_argument("--lfrm-belief-temperature", type=float, default=1.0)
    parser.add_argument("--lfrm-belief-step-size", type=float, default=0.25)
    parser.add_argument("--lfrm-belief-floor", type=float, default=1e-5)
    parser.add_argument("--lfrm-assignment-temperature", type=float, default=1.0)
    parser.add_argument("--lfrm-energy-hidden-dim", type=int, default=128)
    parser.add_argument("--lfrm-use-condition-type-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--flatten-optimizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compute-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from a run checkpoint directory or a concrete step_N checkpoint.",
    )
    parser.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", type=str, default="latent-factor-recurrent-model")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser


def build_config(args: argparse.Namespace, *, vocab_size: int, seq_len: int) -> ExperimentConfig:
    model = ModelConfig(
        vocab_size=vocab_size,
        model_type=args.model_type,
        seq_len=seq_len,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        d_model=args.d_model,
        num_steps=args.num_steps,
        dropout_rate=args.dropout_rate,
        lfrm=LFRMConfig(
            belief_dim=args.lfrm_belief_dim,
            num_slots=args.lfrm_num_slots,
            num_branches=args.lfrm_num_branches,
            num_heads=args.lfrm_num_heads,
            latent_processor_layers=args.lfrm_latent_processor_layers,
            symbol_context_mode=args.lfrm_symbol_context_mode,
            slot_readout_mode=args.lfrm_slot_readout_mode,
            energy_symbol_pooling=args.lfrm_energy_symbol_pooling,
            branch_diversity_schedule=args.lfrm_branch_diversity_schedule,
            diversity_apply_steps=tuple(args.lfrm_diversity_apply_steps),
            belief_temperature=args.lfrm_belief_temperature,
            belief_step_size=args.lfrm_belief_step_size,
            belief_floor=args.lfrm_belief_floor,
            assignment_temperature=args.lfrm_assignment_temperature,
            energy_hidden_dim=args.lfrm_energy_hidden_dim,
            use_condition_type_embedding=args.lfrm_use_condition_type_embedding,
        ),
    )
    optimizer = OptimizerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm,
        flatten_optimizer=args.flatten_optimizer,
    )
    train = TrainConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        step_loss_weighting=args.step_loss_weighting,
        terminal_residual_weight=args.terminal_residual_weight,
        energy_loss_weight=args.energy_loss_weight,
        energy_margin=args.energy_margin,
        energy_corruptions=args.energy_corruptions,
        slot_consistency_weight=args.slot_consistency_weight,
        slot_usage_weight=args.slot_usage_weight,
        slot_diversity_weight=args.slot_diversity_weight,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        ema=EMAConfig(
            enabled=args.ema_enabled,
            decay=args.ema_decay,
        ),
    )
    data = DataConfig(dataset_path=args.dataset_path)
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


def checkpoint_step(path: Path) -> int | None:
    if not path.name.startswith("step_"):
        return None
    try:
        return int(path.name.removeprefix("step_"))
    except ValueError:
        return None


def resolve_resume_checkpoint(resume_from: str) -> tuple[Path, Path]:
    resume_path = Path(resume_from).expanduser().resolve()
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume path does not exist: {resume_path}")
    if checkpoint_step(resume_path) is not None:
        return resume_path, resume_path.parent
    candidates = [
        child
        for child in resume_path.iterdir()
        if child.is_dir() and checkpoint_step(child) is not None
    ]
    if not candidates:
        raise FileNotFoundError(f"No step_N checkpoints found under resume path: {resume_path}")
    checkpoint_path = max(candidates, key=lambda path: checkpoint_step(path) or -1)
    return checkpoint_path, resume_path


def wandb_run_id_path(run_dir: Path) -> Path:
    return run_dir / "wandb_run_id.txt"


def read_wandb_run_id(run_dir: Path) -> str | None:
    explicit_path = wandb_run_id_path(run_dir)
    if explicit_path.exists():
        run_id = explicit_path.read_text(encoding="utf-8").strip()
        return run_id or None
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        return None
    runs = sorted(wandb_dir.glob("run-*-*"), key=lambda path: path.stat().st_mtime, reverse=True)
    for run_path in runs:
        run_id = run_path.name.rsplit("-", 1)[-1]
        if run_id:
            return run_id
    return None


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


def init_wandb(config: ExperimentConfig, *, run_dir: Path, resume_run_id: str | None = None):
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
        id=resume_run_id,
        resume="allow" if resume_run_id is not None else None,
    )
    wandb_run_id_path(run_dir).write_text(run.id + "\n", encoding="utf-8")
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


OPTIONAL_SCALAR_METRICS = (
    "unroll_steps",
    "energy_pos",
    "energy_neg",
    "energy_margin_loss",
    "belief_entropy",
    "belief_confidence",
    "target_probability",
    "branch_min_ce",
    "branch_mean_ce",
    "selected_branch_energy",
    "slot_consistency_loss",
    "slot_usage_entropy",
    "slot_usage_loss",
    "slot_diversity_loss",
    "branch_diversity",
    "final_branch_diversity",
    "terminal_belief_delta",
    "terminal_belief_mse",
)
INTEGER_SCALAR_METRICS = {"unroll_steps"}


def format_scalar_metric(name: str, value: Any) -> str:
    scalar = float(value)
    is_integer_like = np.isfinite(scalar) and np.isclose(scalar, round(scalar), atol=1e-6)
    if name in INTEGER_SCALAR_METRICS or name.endswith(("_count", "_iters", "_steps")):
        if is_integer_like:
            return str(int(round(scalar)))
    if scalar != 0.0 and abs(scalar) < 1e-3:
        return f"{scalar:.2e}"
    return f"{scalar:.4f}"


def optional_scalar_log(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    log: dict[str, float] = {}
    for name in OPTIONAL_SCALAR_METRICS:
        if name in metrics:
            value = metrics[name]
            if np.ndim(np.asarray(value)) == 0:
                log[f"{prefix}/{name}"] = float(value)
    return log


def optional_scalar_summary(metrics: dict[str, Any]) -> str:
    parts = []
    for name in OPTIONAL_SCALAR_METRICS:
        if name in metrics:
            value = metrics[name]
            if np.ndim(np.asarray(value)) == 0:
                parts.append(f"{name}={format_scalar_metric(name, value)}")
    return " ".join(parts)


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

    resume_checkpoint: Path | None = None
    resume_step = 0
    resume_run_id: str | None = None
    if args.resume_from is not None:
        resume_checkpoint, checkpoint_dir = resolve_resume_checkpoint(args.resume_from)
        resume_step = checkpoint_step(resume_checkpoint) or 0
        resume_run_id = read_wandb_run_id(checkpoint_dir)
    else:
        checkpoint_dir = build_run_checkpoint_dir(args.config, config.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(config, run_dir=checkpoint_dir, resume_run_id=resume_run_id)

    model = create_model(config)
    optimizer = create_optimizer(model, config)
    train_step_fn = build_train_step_runner(
        config.train.step_loss_weighting,
        config.train.terminal_residual_weight,
        config.train.energy_loss_weight,
        config.train.energy_margin,
        config.train.energy_corruptions,
        config.train.slot_consistency_weight,
        config.train.slot_usage_weight,
        config.train.slot_diversity_weight,
    )
    eval_step_fn = build_eval_step_runner(
        config.train.step_loss_weighting,
        config.train.terminal_residual_weight,
        config.train.energy_loss_weight,
        config.train.energy_margin,
        config.train.energy_corruptions,
        config.train.slot_consistency_weight,
        config.train.slot_usage_weight,
        config.train.slot_diversity_weight,
    )
    ema_model = create_ema_model(model, config) if config.train.use_ema else None
    ema_update_fn = build_ema_update_runner(config.train.ema_decay) if config.train.use_ema else None
    if resume_checkpoint is not None:
        restored_step = load_checkpoint(resume_checkpoint, model, optimizer, ema_model=ema_model)
        if restored_step != resume_step:
            resume_step = restored_step

    train_rng = np.random.default_rng(config.train.seed + resume_step)
    eval_rng = np.random.default_rng(config.train.seed + 1 + resume_step)
    train_key = jax.random.fold_in(jax.random.key(config.train.seed), resume_step)

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
        "resume_checkpoint=", resume_checkpoint,
        "resume_step=", resume_step,
    )

    current_batch = sample_device_batch(
        train_rng,
        dataset,
        config=config,
        split="train",
        device=device,
    )

    for step in range(resume_step + 1, config.train.max_steps + 1):
        train_key, step_key = jax.random.split(train_key)
        metrics = train_step_fn(
            model,
            optimizer,
            current_batch,
            step_key,
        )
        if ema_model is not None and ema_update_fn is not None:
            ema_update_fn(ema_model, model)

        current_batch = sample_device_batch(
            train_rng,
            dataset,
            config=config,
            split="train",
            device=device,
        )

        if step % config.train.log_every == 0 or step == 1:
            train_log = {
                "train/loss": float(metrics["loss"]),
                "train/blank_ce_loss": float(metrics["blank_ce_loss"]),
                "train/final_blank_ce_loss": float(metrics["final_blank_ce_loss"]),
                "train/blank_cell_accuracy": float(metrics["blank_cell_accuracy"]),
                "train/solved_rate": float(metrics["solved_rate"]),
                "train/learning_rate": schedule_learning_rate(config, step),
            }
            train_log.update(optional_scalar_log("train", metrics))
            if wandb_run is not None:
                wandb_run.log(train_log, step=step)
            optional_summary = optional_scalar_summary(metrics)
            print(
                f"step={step} "
                f"loss={float(metrics['loss']):.4f} "
                f"ce={float(metrics['blank_ce_loss']):.4f} "
                f"final_ce={float(metrics['final_blank_ce_loss']):.4f} "
                f"blank_acc={float(metrics['blank_cell_accuracy']):.4f} "
                f"solved={float(metrics['solved_rate']):.4f}"
                f"{' ' + optional_summary if optional_summary else ''}"
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
                    "eval/final_blank_ce_loss": eval_metrics["final_blank_ce_loss"],
                    "eval/blank_cell_accuracy": eval_metrics["blank_cell_accuracy"],
                    "eval/solved_rate": eval_metrics["solved_rate"],
                }
                eval_log.update(optional_scalar_log("eval", eval_metrics))
                eval_log.update(flatten_step_metrics("eval/loss_by_step", eval_metrics["per_step_loss"]))
                eval_log.update(
                    flatten_step_metrics(
                        "eval/hidden_delta_by_step",
                        eval_metrics["per_step_hidden_delta"],
                    )
                )
                wandb_run.log(eval_log, step=step)
            optional_summary = optional_scalar_summary(eval_metrics)
            print(
                f"[eval{'/ema' if ema_model is not None else ''}] step={step} "
                f"loss={eval_metrics['loss']:.4f} "
                f"ce={eval_metrics['blank_ce_loss']:.4f} "
                f"final_ce={eval_metrics['final_blank_ce_loss']:.4f} "
                f"blank_acc={eval_metrics['blank_cell_accuracy']:.4f} "
                f"solved={eval_metrics['solved_rate']:.4f}"
                f"{' ' + optional_summary if optional_summary else ''}"
            )
            print(
                "  "
                + " ".join(
                    [
                        format_step_summary("loss", eval_metrics["per_step_loss"]),
                        format_step_summary("delta", eval_metrics["per_step_hidden_delta"]),
                    ]
                )
            )
            save_checkpoint(str(checkpoint_dir), model, optimizer, step, ema_model=ema_model)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
