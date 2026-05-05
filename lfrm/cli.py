from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
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
    TRMConfig,
    TrainConfig,
    WandbConfig,
)
from lfrm.datasets import dataset_overview, load_dataset, sample_batch
from lfrm.training import (
    build_ema_update_runner,
    build_eval_step_runner,
    build_train_step_runner,
    build_trm_act_train_step_runner,
    build_trm_dense_unroll_train_step_runner,
    build_trm_eval_step_runner,
    create_ema_model,
    create_model,
    create_optimizer,
    ema_param_filter,
    load_checkpoint,
    save_checkpoint,
)


CONFIG_SECTIONS = ("data", "model", "optimizer", "train", "runtime", "wandb")
NESTED_SECTIONS = {
    "model": {"lfrm", "trm"},
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
        "trm",
    },
    "optimizer": {
        "learning_rate",
        "lr_min_ratio",
        "beta1",
        "beta2",
        "weight_decay",
        "warmup_steps",
        "grad_clip_norm",
        "flatten_optimizer",
    },
    "train": {
        "batch_size",
        "eval_batch_size",
        "max_steps",
        "log_every",
        "eval_every",
        "eval_batches",
        "step_loss_weighting",
        "trm_train_mode",
        "dense_loss_weight",
        "final_loss_weight",
        "q_loss_weight",
        "terminal_residual_weight",
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
        "num_heads",
        "latent_processor_layers",
        "symbol_context_mode",
        "slot_readout_mode",
        "belief_temperature",
        "belief_step_size",
        "belief_floor",
        "assignment_temperature",
        "use_condition_type_embedding",
    },
    "ema": {"enabled", "decay"},
    "trm": {
        "h_cycles",
        "l_cycles",
        "l_layers",
        "num_heads",
        "mlp_ratio",
        "mlp_t",
        "puzzle_emb_ndim",
        "puzzle_emb_len",
        "pos_encodings",
        "rms_norm_eps",
        "rope_theta",
        "halt_exploration_prob",
        "no_act_continue",
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
        allowed_keys = ALLOWED_SECTION_KEYS[section]
        for key, value in section_values.items():
            normalized_key = key.replace("-", "_")
            if normalized_key not in allowed_keys:
                raise ValueError(f"Unsupported [{section}] field in grid reasoning config: {key}")
            if normalized_key in NESTED_SECTIONS.get(section, set()):
                if not isinstance(value, dict):
                    raise ValueError(f"Section [{section}.{key}] in {config_path} must be a table")
                for nested_key, nested_value in value.items():
                    normalized_nested_key = nested_key.replace("-", "_")
                    if normalized_nested_key not in ALLOWED_NESTED_KEYS[normalized_key]:
                        raise ValueError(f"Unsupported [{section}.{key}] field in grid reasoning config: {nested_key}")
                    flat[f"{normalized_key}_{normalized_nested_key}"] = nested_value
                continue
            if section == "wandb":
                normalized_key = f"wandb_{normalized_key}"
            flat[normalized_key] = value

    if flat.get("model_type", "lfrm") not in ("lfrm", "trm"):
        raise ValueError("Only model_type='lfrm' or model_type='trm' is supported")
    return flat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train recurrent grid reasoning models.")
    parser.add_argument("--config", type=str, default=None, help="Optional TOML config file.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Offline grid dataset directory.")
    parser.add_argument("--seq-len", type=int, default=81)
    parser.add_argument("--grid-height", type=int, default=9)
    parser.add_argument("--grid-width", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Eval batch size. Uses --batch-size when set to 0.",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--step-loss-weighting",
        choices=("uniform", "linear", "final"),
        default="uniform",
        help="How to weight losses from recurrent reasoning steps.",
    )
    parser.add_argument(
        "--trm-train-mode",
        choices=("act", "dense_unroll"),
        default="act",
        help="TRM training path: ACT single-step carry or full-unroll dense CE.",
    )
    parser.add_argument("--dense-loss-weight", type=float, default=0.5)
    parser.add_argument("--final-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--q-loss-weight", type=float, default=0.0)
    parser.add_argument("--terminal-residual-weight", type=float, default=0.0)
    parser.add_argument("--slot-consistency-weight", type=float, default=0.0)
    parser.add_argument("--slot-usage-weight", type=float, default=0.0)
    parser.add_argument("--slot-diversity-weight", type=float, default=0.0)
    parser.add_argument("--ema-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--model-type", choices=("lfrm", "trm"), default="lfrm")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=6)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--lfrm-belief-dim", type=int, default=0)
    parser.add_argument("--lfrm-num-slots", type=int, default=64)
    parser.add_argument("--lfrm-num-heads", type=int, default=4)
    parser.add_argument("--lfrm-latent-processor-layers", type=int, default=1)
    parser.add_argument("--lfrm-symbol-context-mode", choices=("cell_symbol_tokens",), default="cell_symbol_tokens")
    parser.add_argument("--lfrm-slot-readout-mode", choices=("cell_symbol_attention",), default="cell_symbol_attention")
    parser.add_argument("--lfrm-belief-temperature", type=float, default=1.0)
    parser.add_argument("--lfrm-belief-step-size", type=float, default=0.25)
    parser.add_argument("--lfrm-belief-floor", type=float, default=1e-5)
    parser.add_argument("--lfrm-assignment-temperature", type=float, default=1.0)
    parser.add_argument("--lfrm-use-condition-type-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trm-h-cycles", type=int, default=3)
    parser.add_argument("--trm-l-cycles", type=int, default=6)
    parser.add_argument("--trm-l-layers", type=int, default=2)
    parser.add_argument("--trm-num-heads", type=int, default=8)
    parser.add_argument("--trm-mlp-ratio", type=int, default=4)
    parser.add_argument("--trm-mlp-t", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trm-puzzle-emb-ndim", type=int, default=0)
    parser.add_argument("--trm-puzzle-emb-len", type=int, default=16)
    parser.add_argument("--trm-pos-encodings", choices=("none", "learned", "rope", "grid"), default="none")
    parser.add_argument("--trm-rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--trm-rope-theta", type=float, default=10000.0)
    parser.add_argument("--trm-halt-exploration-prob", type=float, default=0.1)
    parser.add_argument("--trm-no-act-continue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
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


def build_config(
    args: argparse.Namespace,
    *,
    vocab_size: int,
    num_puzzle_identifiers: int,
    seq_len: int,
) -> ExperimentConfig:
    model = ModelConfig(
        vocab_size=vocab_size,
        num_puzzle_identifiers=num_puzzle_identifiers,
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
            num_heads=args.lfrm_num_heads,
            latent_processor_layers=args.lfrm_latent_processor_layers,
            symbol_context_mode=args.lfrm_symbol_context_mode,
            slot_readout_mode=args.lfrm_slot_readout_mode,
            belief_temperature=args.lfrm_belief_temperature,
            belief_step_size=args.lfrm_belief_step_size,
            belief_floor=args.lfrm_belief_floor,
            assignment_temperature=args.lfrm_assignment_temperature,
            use_condition_type_embedding=args.lfrm_use_condition_type_embedding,
        ),
        trm=TRMConfig(
            h_cycles=args.trm_h_cycles,
            l_cycles=args.trm_l_cycles,
            l_layers=args.trm_l_layers,
            num_heads=args.trm_num_heads,
            mlp_ratio=args.trm_mlp_ratio,
            mlp_t=args.trm_mlp_t,
            puzzle_emb_ndim=args.trm_puzzle_emb_ndim,
            puzzle_emb_len=args.trm_puzzle_emb_len,
            pos_encodings=args.trm_pos_encodings,
            rms_norm_eps=args.trm_rms_norm_eps,
            rope_theta=args.trm_rope_theta,
            halt_exploration_prob=args.trm_halt_exploration_prob,
            no_act_continue=args.trm_no_act_continue,
        ),
    )
    optimizer = OptimizerConfig(
        learning_rate=args.learning_rate,
        lr_min_ratio=args.lr_min_ratio,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm,
        flatten_optimizer=args.flatten_optimizer,
    )
    train = TrainConfig(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_steps=args.max_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        step_loss_weighting=args.step_loss_weighting,
        trm_train_mode=args.trm_train_mode,
        dense_loss_weight=args.dense_loss_weight,
        final_loss_weight=args.final_loss_weight,
        q_loss_weight=args.q_loss_weight,
        terminal_residual_weight=args.terminal_residual_weight,
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
    optimizer_step = max(1, step)
    optimizer_steps = max(1, config.train.max_steps)
    warmup_steps = max(1, config.optimizer.warmup_steps)
    peak = config.optimizer.learning_rate
    end = peak * config.optimizer.lr_min_ratio
    decay_steps = max(optimizer_steps, warmup_steps + 1)
    if optimizer_step <= warmup_steps:
        return peak * optimizer_step / max(warmup_steps, 1)
    progress = min(max((optimizer_step - warmup_steps) / max(decay_steps - warmup_steps, 1), 0.0), 1.0)
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


class BatchPrefetcher:
    def __init__(self, sample_fn, *, depth: int = 2) -> None:
        if depth < 1:
            raise ValueError("Prefetch depth must be at least 1")
        self.sample_fn = sample_fn
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.futures: list[Future] = []
        for _ in range(depth):
            self.futures.append(self.executor.submit(self.sample_fn))

    def next(self):
        future = self.futures.pop(0)
        batch = future.result()
        self.futures.append(self.executor.submit(self.sample_fn))
        return batch

    def close(self) -> None:
        for future in self.futures:
            future.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)


def eval_device_batch(
    dataset,
    *,
    config: ExperimentConfig,
    start: int,
    stop: int,
    device: jax.Device,
) -> dict[str, jax.Array]:
    if dataset.spec.seq_len != config.model.seq_len:
        raise ValueError(f"Requested seq_len={config.model.seq_len}, but dataset seq_len={dataset.spec.seq_len}")
    batch = {
        "inputs": np.asarray(dataset.eval_inputs[start:stop], dtype=np.int32),
        "labels": np.asarray(dataset.eval_labels[start:stop], dtype=np.int32),
        "given_mask": np.asarray(dataset.eval_given_mask[start:stop], dtype=bool),
        "puzzle_identifiers": np.asarray(dataset.eval_puzzle_identifiers[start:stop], dtype=np.int32),
    }
    return jax.device_put(batch, device=device)


def eval_device_index_batch(
    dataset,
    *,
    config: ExperimentConfig,
    indices: np.ndarray,
    device: jax.Device,
) -> dict[str, jax.Array]:
    if dataset.spec.seq_len != config.model.seq_len:
        raise ValueError(f"Requested seq_len={config.model.seq_len}, but dataset seq_len={dataset.spec.seq_len}")
    batch = {
        "inputs": np.asarray(dataset.eval_inputs[indices], dtype=np.int32),
        "labels": np.asarray(dataset.eval_labels[indices], dtype=np.int32),
        "given_mask": np.asarray(dataset.eval_given_mask[indices], dtype=bool),
        "puzzle_identifiers": np.asarray(dataset.eval_puzzle_identifiers[indices], dtype=np.int32),
    }
    return jax.device_put(batch, device=device)


def evaluate(eval_step_fn, model, dataset, *, config: ExperimentConfig, rng: np.random.Generator) -> dict[str, Any]:
    device = jax.devices()[0]
    reduced: dict[str, Any] | None = None
    total = dataset.eval_inputs.shape[0]
    if total == 0:
        raise ValueError("Eval split is empty")
    batch_size = config.train.eval_batch_size or config.train.batch_size
    if batch_size <= 0:
        raise ValueError("eval_batch_size must be at least 1 when set")
    total_weight = 0.0
    if config.train.eval_batches <= 0:
        batches = [
            ("slice", start, min(start + batch_size, total))
            for start in range(0, total, batch_size)
        ]
    else:
        sample_count = min(total, config.train.eval_batches * batch_size)
        indices = rng.choice(total, size=sample_count, replace=False)
        batches = [
            ("indices", start, min(start + batch_size, sample_count))
            for start in range(0, sample_count, batch_size)
        ]

    print(
        f"[eval] running {len(batches)} batches "
        f"x batch_size={batch_size} "
        f"({sum(stop - start for _mode, start, stop in batches)} examples)",
        flush=True,
    )
    for batch_index, (mode, start, stop) in enumerate(batches, start=1):
        if mode == "slice":
            batch = eval_device_batch(
                dataset,
                config=config,
                start=start,
                stop=stop,
                device=device,
            )
        else:
            batch = eval_device_index_batch(
                dataset,
                config=config,
                indices=indices[start:stop],
                device=device,
            )
        metrics = eval_step_fn(model, batch)
        weight = float(stop - start)
        if reduced is None:
            reduced = {
                key: np.zeros(np.asarray(value).shape, dtype=np.float64)
                for key, value in metrics.items()
            }
        for key, value in metrics.items():
            value_array = np.asarray(jax.device_get(value), dtype=np.float64)
            if key.endswith("_count"):
                reduced[key] += value_array
            else:
                reduced[key] += value_array * weight
        total_weight += weight
        if batch_index == 1 or batch_index == len(batches) or batch_index % 10 == 0:
            print(f"[eval] batch {batch_index}/{len(batches)}", flush=True)
    if reduced is None:
        raise ValueError("No eval batches were produced")
    scale = 1.0 / total_weight
    averaged = {
        key: value if key.endswith("_count") else value * scale
        for key, value in reduced.items()
    }
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


CORE_SCALAR_METRICS = (
    "unroll_steps",
    "belief_entropy",
    "belief_confidence",
    "target_probability",
    "mean_blank_ce_loss",
    "q_loss",
    "q_selected_blank_ce_loss",
    "q_selected_blank_cell_accuracy",
    "q_selected_solved_rate",
    "q_selected_solved_count",
    "q_selected_step",
    "oracle_step",
    "act_step",
    "halted_rate",
    "reset_rate",
    "solved_count",
    "final_blank_cell_accuracy",
    "final_solved_rate",
    "final_solved_count",
)
SLOT_DIAGNOSTIC_METRICS = (
    "slot_consistency_loss",
    "slot_usage_entropy",
    "slot_usage_loss",
    "slot_diversity_loss",
)
TERMINAL_DIAGNOSTIC_METRICS = (
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


def scalar_metric_names(config: ExperimentConfig) -> tuple[str, ...]:
    names = list(CORE_SCALAR_METRICS)
    if (
        config.train.slot_consistency_weight != 0.0
        or config.train.slot_usage_weight != 0.0
        or config.train.slot_diversity_weight != 0.0
    ):
        names.extend(SLOT_DIAGNOSTIC_METRICS)
    if config.train.terminal_residual_weight != 0.0:
        names.extend(TERMINAL_DIAGNOSTIC_METRICS)
    return tuple(names)


def optional_scalar_log(prefix: str, metrics: dict[str, Any], names: tuple[str, ...]) -> dict[str, float]:
    log: dict[str, float] = {}
    for name in names:
        if name in metrics:
            value = metrics[name]
            if np.ndim(np.asarray(value)) == 0:
                log[f"{prefix}/{name}"] = float(value)
    return log


def optional_scalar_summary(metrics: dict[str, Any], names: tuple[str, ...]) -> str:
    parts = []
    for name in names:
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
        num_puzzle_identifiers=dataset.spec.num_puzzle_identifiers,
        seq_len=dataset.spec.seq_len,
    )
    if config.train.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.train.eval_batch_size < 0:
        raise ValueError("eval_batch_size must be non-negative")
    if config.model.model_type != "trm" and config.train.trm_train_mode != "act":
        raise ValueError("trm_train_mode is only supported for model_type='trm'")
    if config.train.dense_loss_weight < 0.0 or config.train.final_loss_weight < 0.0:
        raise ValueError("dense_loss_weight and final_loss_weight must be non-negative")
    if config.train.trm_train_mode == "dense_unroll" and (
        config.train.dense_loss_weight + config.train.final_loss_weight <= 0.0
    ):
        raise ValueError("dense_unroll requires a positive dense or final loss weight")

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
    if config.model.model_type == "trm":
        if config.train.trm_train_mode == "dense_unroll":
            train_step_fn = build_trm_dense_unroll_train_step_runner(
                dense_loss_weight=config.train.dense_loss_weight,
                final_loss_weight=config.train.final_loss_weight,
                q_loss_weight=config.train.q_loss_weight,
            )
        else:
            train_step_fn = build_trm_act_train_step_runner(config.train.q_loss_weight)
        eval_step_fn = build_trm_eval_step_runner(config.train.q_loss_weight)
    else:
        train_step_fn = build_train_step_runner(
            config.train.step_loss_weighting,
            config.train.q_loss_weight,
            config.train.terminal_residual_weight,
            config.train.slot_consistency_weight,
            config.train.slot_usage_weight,
            config.train.slot_diversity_weight,
        )
        eval_step_fn = build_eval_step_runner(
            config.train.step_loss_weighting,
            config.train.q_loss_weight,
            config.train.terminal_residual_weight,
            config.train.slot_consistency_weight,
            config.train.slot_usage_weight,
            config.train.slot_diversity_weight,
        )
    ema_model = create_ema_model(model, config) if config.train.use_ema else None
    ema_update_fn = (
        build_ema_update_runner(config.train.ema_decay, ema_param_filter(config))
        if config.train.use_ema
        else None
    )
    if resume_checkpoint is not None:
        restored_step = load_checkpoint(resume_checkpoint, model, optimizer, ema_model=ema_model)
        if restored_step != resume_step:
            resume_step = restored_step

    train_rng = np.random.default_rng(config.train.seed + resume_step)
    eval_rng = np.random.default_rng(config.train.seed + 1 + resume_step)
    train_key = jax.random.fold_in(jax.random.key(config.train.seed), resume_step)
    scalar_metrics = scalar_metric_names(config)

    device = jax.devices()[0]
    overview = dataset_overview(dataset)
    print(
        "device=", device,
        "dataset_kind=", overview["kind"],
        "task_type=", overview["task_type"],
        "vocab_size=", overview["vocab_size"],
        "num_puzzle_identifiers=", overview["num_puzzle_identifiers"],
        "train_examples=", overview["train_examples"],
        "eval_examples=", overview["eval_examples"],
        "batch_size=", config.train.batch_size,
        "eval_batch_size=", config.train.eval_batch_size or config.train.batch_size,
        "trm_train_mode=", config.train.trm_train_mode,
        "seq_len=", config.model.seq_len,
        "grid_height=", config.model.grid_height,
        "grid_width=", config.model.grid_width,
        "checkpoint_dir=", checkpoint_dir,
        "resume_checkpoint=", resume_checkpoint,
        "resume_step=", resume_step,
    )

    def sample_train_batch():
        return sample_device_batch(
            train_rng,
            dataset,
            config=config,
            split="train",
            device=device,
        )

    prefetcher = BatchPrefetcher(sample_train_batch, depth=2)

    current_batch = prefetcher.next()
    use_trm_act = config.model.model_type == "trm" and config.train.trm_train_mode == "act"
    train_carry = model.initial_carry(current_batch) if use_trm_act else None

    try:
        for step in range(resume_step + 1, config.train.max_steps + 1):
            is_eval_step = step % config.train.eval_every == 0 or step == config.train.max_steps
            train_key, step_key = jax.random.split(train_key)
            if use_trm_act:
                metrics, train_carry = train_step_fn(
                    model,
                    optimizer,
                    train_carry,
                    current_batch,
                    step_key,
                )
            else:
                metrics = train_step_fn(
                    model,
                    optimizer,
                    current_batch,
                    step_key,
                )
            current_batch = prefetcher.next()

            if ema_model is not None and ema_update_fn is not None:
                ema_update_fn(ema_model, model)

            if step % config.train.log_every == 0 or step == 1:
                train_log = {
                    "train/loss": float(metrics["loss"]),
                    "train/blank_ce_loss": float(metrics["blank_ce_loss"]),
                    "train/final_blank_ce_loss": float(metrics["final_blank_ce_loss"]),
                    "train/blank_cell_accuracy": float(metrics["blank_cell_accuracy"]),
                    "train/solved_rate": float(metrics["solved_rate"]),
                    "train/learning_rate": schedule_learning_rate(config, step),
                }
                train_log.update(optional_scalar_log("train", metrics, scalar_metrics))
                if "per_step_loss" in metrics:
                    train_log.update(
                        flatten_step_metrics(
                            "train/loss_by_step",
                            list(jax.device_get(metrics["per_step_loss"])),
                        )
                    )
                if wandb_run is not None:
                    wandb_run.log(train_log, step=step, commit=not is_eval_step)
                optional_summary = optional_scalar_summary(metrics, scalar_metrics)
                print(
                    f"step={step} "
                    f"loss={float(metrics['loss']):.4f} "
                    f"ce={float(metrics['blank_ce_loss']):.4f} "
                    f"final_ce={float(metrics['final_blank_ce_loss']):.4f} "
                    f"blank_acc={float(metrics['blank_cell_accuracy']):.4f} "
                    f"solved={float(metrics['solved_rate']):.4f}"
                    f"{' ' + optional_summary if optional_summary else ''}"
                )

            if is_eval_step:
                def run_eval_and_log(eval_model, prefix: str, label: str, *, commit: bool) -> dict[str, Any]:
                    eval_metrics = evaluate(
                        eval_step_fn,
                        eval_model,
                        dataset,
                        config=config,
                        rng=eval_rng,
                    )
                    if wandb_run is not None:
                        eval_log = {
                            f"{prefix}/loss": eval_metrics["loss"],
                            f"{prefix}/blank_ce_loss": eval_metrics["blank_ce_loss"],
                            f"{prefix}/final_blank_ce_loss": eval_metrics["final_blank_ce_loss"],
                            f"{prefix}/final_blank_cell_accuracy": eval_metrics["final_blank_cell_accuracy"],
                            f"{prefix}/final_solved_rate": eval_metrics["final_solved_rate"],
                            f"{prefix}/blank_cell_accuracy": eval_metrics["blank_cell_accuracy"],
                            f"{prefix}/solved_rate": eval_metrics["solved_rate"],
                        }
                        eval_log.update(optional_scalar_log(prefix, eval_metrics, scalar_metrics))
                        eval_log.update(flatten_step_metrics(f"{prefix}/loss_by_step", eval_metrics["per_step_loss"]))
                        eval_log.update(
                            flatten_step_metrics(
                                f"{prefix}/hidden_delta_by_step",
                                eval_metrics["per_step_hidden_delta"],
                            )
                        )
                        if "per_step_quality_score" in eval_metrics:
                            eval_log.update(
                                flatten_step_metrics(
                                    f"{prefix}/quality_by_step",
                                    eval_metrics["per_step_quality_score"],
                                )
                            )
                        wandb_run.log(eval_log, step=step, commit=commit)
                    optional_summary = optional_scalar_summary(eval_metrics, scalar_metrics)
                    print(
                        f"[{label}] step={step} "
                        f"loss={eval_metrics['loss']:.4f} "
                        f"ce={eval_metrics['blank_ce_loss']:.4f} "
                        f"final_ce={eval_metrics['final_blank_ce_loss']:.4f} "
                        f"blank_acc={eval_metrics['blank_cell_accuracy']:.4f} "
                        f"final_acc={eval_metrics['final_blank_cell_accuracy']:.4f} "
                        f"solved={eval_metrics['solved_rate']:.4f}"
                        f"{' ' + optional_summary if optional_summary else ''}"
                    )
                    print(
                        "  "
                        + " ".join(
                            [
                                format_step_summary("loss", eval_metrics["per_step_loss"]),
                                format_step_summary("delta", eval_metrics["per_step_hidden_delta"]),
                                format_step_summary("q", eval_metrics.get("per_step_quality_score", [])),
                            ]
                        )
                    )
                    return eval_metrics

                if ema_model is not None:
                    run_eval_and_log(model, "eval/raw", "eval/raw", commit=False)
                    run_eval_and_log(ema_model, "eval/ema", "eval/ema", commit=True)
                else:
                    run_eval_and_log(model, "eval", "eval", commit=True)
                save_checkpoint(str(checkpoint_dir), model, optimizer, step, ema_model=ema_model)
    finally:
        with suppress(Exception):
            prefetcher.close()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
