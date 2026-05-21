from __future__ import annotations

from lfrm.jax_defaults import apply_jax_defaults

apply_jax_defaults()

import argparse
from pathlib import Path
import tomllib

from lfrm.config import (
    DataConfig,
    EMAConfig,
    ExperimentConfig,
    EvalConfig,
    BRCConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    TaskConfig,
    TRMConfig,
    TrainConfig,
    URMConfig,
    WandbConfig,
)
from lfrm.datasets import load_dataset
from lfrm.runtime import (
    apply_epoch_budget,
    run_training,
)


CONFIG_SECTIONS = ("data", "task", "model", "optimizer", "train", "eval", "runtime", "wandb")
NESTED_SECTIONS = {
    "model": {"trm", "brc", "urm"},
    "train": {"ema"},
}
ALLOWED_SECTION_KEYS = {
    "data": {"dataset_path"},
    "task": {"type"},
    "model": {
        "model_type",
        "seq_len",
        "grid_height",
        "grid_width",
        "d_model",
        "rollout_steps",
        "dropout_rate",
        "loss_type",
        "trm",
        "brc",
        "urm",
    },
    "optimizer": {
        "optimizer_type",
        "learning_rate",
        "puzzle_embed_learning_rate",
        "lr_min_ratio",
        "beta1",
        "beta2",
        "weight_decay",
        "puzzle_embed_weight_decay",
        "puzzle_embed_coalesce_updates",
        "lr_warmup_steps",
        "grad_clip_norm",
        "flatten_optimizer",
    },
    "train": {
        "batch_size",
        "epochs",
        "log_epochs",
        "trm_train_mode",
        "halt_loss_weight",
        "terminal_residual_weight",
        "seed",
        "checkpoint_dir",
        "ema",
    },
    "eval": {
        "batch_size",
        "epochs",
        "diagnostics",
        "full_dataset",
    },
    "runtime": {
        "compute_dtype",
        "data_parallel_devices",
        "prefetch_depth",
        "prefetch_workers",
        "train_dispatch_chunk",
        "profile_enabled",
        "profile_start_step",
        "profile_steps",
        "profile_dir",
    },
    "wandb": {"enabled", "project", "entity", "name", "mode"},
}
ALLOWED_NESTED_KEYS = {
    "ema": {"enabled", "decay"},
    "trm": {
        "deep_recursion",
        "latent_recursion",
        "block_layers",
        "num_heads",
        "mlp_ratio",
        "mlp_t",
        "local_mixing",
        "local_mixing_kernel",
        "puzzle_embed_ndim",
        "puzzle_embed_len",
        "position_encoding",
        "rms_norm_eps",
        "rope_theta",
        "halt_exploration_prob",
        "no_act_continue",
        "step_loss_weights",
    },
    "brc": {
        "recurrent_steps",
        "block_layers",
        "num_heads",
        "mlp_ratio",
        "position_encoding",
        "rms_norm_eps",
        "rope_theta",
        "step_loss_weights",
        "denoise_initial_prob",
        "denoise_trajectory_prob",
        "denoise_teacher_reveal_prob",
        "denoise_mode_weights",
        "fixed_point_entropy_weight",
        "fixed_point_loss_weight",
        "context_weight_reg_weight",
    },
    "urm": {
        "recurrent_steps",
        "deep_recursion",
        "latent_recursion",
        "block_layers",
        "num_heads",
        "mlp_ratio",
        "conv_kernel",
        "puzzle_embed_ndim",
        "puzzle_embed_len",
        "rms_norm_eps",
        "rope_theta",
        "halt_exploration_prob",
        "step_loss_weights",
    },
}


def load_toml_config(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open("rb") as f:
        loaded = tomllib.load(f)

    for section in loaded:
        if section not in CONFIG_SECTIONS:
            raise ValueError(f"Unsupported grid reasoning config section: [{section}]")

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
            if section == "eval":
                normalized_key = f"eval_{normalized_key}"
            if section == "task" and normalized_key == "type":
                normalized_key = "task_type"
            flat[normalized_key] = value

    if flat.get("model_type", "brc") not in ("trm", "brc", "urm"):
        raise ValueError("Only model_type=trm, brc, or urm is supported")
    if flat.get("task_type", "sudoku") not in ("sudoku", "maze", "arc"):
        raise ValueError("Only task_type='sudoku', 'maze', or 'arc' is supported")
    if flat.get("loss_type", "softmax") not in ("softmax", "stablemax"):
        raise ValueError("Only loss_type='softmax' or loss_type='stablemax' is supported")
    if flat.get("optimizer_type", "adamw") not in ("adamw", "adam_atan2", "muon"):
        raise ValueError("Only optimizer_type='adamw', optimizer_type='adam_atan2', or optimizer_type='muon' is supported")
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Grouped dataset epochs. The training loop derives optimizer updates internally.",
    )
    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=100,
        help="Eval interval in grouped dataset epochs. The loop derives optimizer-update intervals internally.",
    )
    parser.add_argument("--eval-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-full-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-epochs", type=int, default=10)
    parser.add_argument(
        "--trm-train-mode",
        choices=("act", "dense_unroll"),
        default="act",
        help="TRM training path: ACT single-step carry or full-unroll dense CE.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--halt-loss-weight", type=float, default=0.0)
    parser.add_argument("--terminal-residual-weight", type=float, default=0.0)
    parser.add_argument("--ema-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--model-type", choices=("trm", "brc", "urm"), default="brc")
    parser.add_argument("--task-type", choices=("sudoku", "maze", "arc"), default="sudoku")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--loss-type", choices=("softmax", "stablemax"), default="softmax")
    parser.add_argument("--trm-deep-recursion", type=int, default=3)
    parser.add_argument("--trm-latent-recursion", type=int, default=6)
    parser.add_argument("--trm-block-layers", type=int, default=2)
    parser.add_argument("--trm-num-heads", type=int, default=8)
    parser.add_argument("--trm-mlp-ratio", type=int, default=4)
    parser.add_argument("--trm-mlp-t", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trm-local-mixing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trm-local-mixing-kernel", type=int, default=3)
    parser.add_argument("--trm-puzzle-embed-ndim", type=int, default=0)
    parser.add_argument("--trm-puzzle-embed-len", type=int, default=16)
    parser.add_argument("--trm-position-encoding", choices=("none", "learned", "rope", "grid"), default="none")
    parser.add_argument("--trm-rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--trm-rope-theta", type=float, default=10000.0)
    parser.add_argument("--trm-halt-exploration-prob", type=float, default=0.1)
    parser.add_argument("--trm-no-act-continue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trm-step-loss-weights", type=float, nargs="*", default=None)
    parser.add_argument("--brc-recurrent-steps", type=int, default=6)
    parser.add_argument("--brc-block-layers", type=int, default=1)
    parser.add_argument("--brc-num-heads", type=int, default=4)
    parser.add_argument("--brc-mlp-ratio", type=int, default=2)
    parser.add_argument("--brc-position-encoding", choices=("rope", "learned", "none"), default="rope")
    parser.add_argument("--brc-rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--brc-rope-theta", type=float, default=10000.0)
    parser.add_argument("--brc-step-loss-weights", type=float, nargs="*", default=None)
    parser.add_argument("--brc-denoise-initial-prob", type=float, default=0.4)
    parser.add_argument("--brc-denoise-trajectory-prob", type=float, default=0.0)
    parser.add_argument("--brc-denoise-teacher-reveal-prob", type=float, default=0.25)
    parser.add_argument("--brc-denoise-mode-weights", type=float, nargs="*", default=None)
    parser.add_argument("--brc-fixed-point-entropy-weight", type=float, default=0.01)
    parser.add_argument("--brc-fixed-point-loss-weight", type=float, default=0.0)
    parser.add_argument("--brc-context-weight-reg-weight", type=float, default=0.0)
    parser.add_argument("--urm-recurrent-steps", type=int, default=16)
    parser.add_argument("--urm-deep-recursion", type=int, default=2)
    parser.add_argument("--urm-latent-recursion", type=int, default=6)
    parser.add_argument("--urm-block-layers", type=int, default=4)
    parser.add_argument("--urm-num-heads", type=int, default=8)
    parser.add_argument("--urm-mlp-ratio", type=int, default=4)
    parser.add_argument("--urm-conv-kernel", type=int, default=2)
    parser.add_argument("--urm-puzzle-embed-ndim", type=int, default=512)
    parser.add_argument("--urm-puzzle-embed-len", type=int, default=1)
    parser.add_argument("--urm-rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--urm-rope-theta", type=float, default=10000.0)
    parser.add_argument("--urm-halt-exploration-prob", type=float, default=0.1)
    parser.add_argument("--urm-step-loss-weights", type=float, nargs="*", default=None)
    parser.add_argument("--optimizer-type", choices=("adamw", "adam_atan2", "muon"), default="adamw")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--puzzle-embed-learning-rate", type=float, default=0.0)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--puzzle-embed-weight-decay", type=float, default=0.0)
    parser.add_argument("--puzzle-embed-coalesce-updates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--flatten-optimizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compute-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--data-parallel-devices",
        type=int,
        default=1,
        help="Number of local devices for data parallelism. Use 0 for all visible devices.",
    )
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=4,
        help="Number of device batches to keep queued ahead of the training loop.",
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=2,
        help="Number of background workers used for batch sampling and device placement.",
    )
    parser.add_argument(
        "--train-dispatch-chunk",
        type=int,
        default=1,
        help=(
            "Reserved optimizer-update chunk size for future JAX scan dispatch. "
            "Currently only train_dispatch_chunk=1 is supported."
        ),
    )
    parser.add_argument(
        "--profile-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture one JAX profiler trace. Disabled by default for stable long runs.",
    )
    parser.add_argument(
        "--profile-start-step",
        type=int,
        default=1000,
        help="Optimizer step at which to start the default JAX profiler trace.",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=20,
        help="Number of optimizer steps to capture in the default JAX profiler trace.",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="profile",
        help="Profile directory relative to the checkpoint run directory, or an absolute path.",
    )
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
    task = TaskConfig(
        type=args.task_type,
    )
    model = ModelConfig(
        vocab_size=vocab_size,
        num_puzzle_identifiers=num_puzzle_identifiers,
        model_type=args.model_type,
        seq_len=seq_len,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        d_model=args.d_model,
        rollout_steps=args.rollout_steps,
        dropout_rate=args.dropout_rate,
        loss_type=args.loss_type,
        task=task,
        trm=TRMConfig(
            deep_recursion=args.trm_deep_recursion,
            latent_recursion=args.trm_latent_recursion,
            block_layers=args.trm_block_layers,
            num_heads=args.trm_num_heads,
            mlp_ratio=args.trm_mlp_ratio,
            mlp_t=args.trm_mlp_t,
            local_mixing=args.trm_local_mixing,
            local_mixing_kernel=args.trm_local_mixing_kernel,
            puzzle_embed_ndim=args.trm_puzzle_embed_ndim,
            puzzle_embed_len=args.trm_puzzle_embed_len,
            position_encoding=args.trm_position_encoding,
            rms_norm_eps=args.trm_rms_norm_eps,
            rope_theta=args.trm_rope_theta,
            halt_exploration_prob=args.trm_halt_exploration_prob,
            no_act_continue=args.trm_no_act_continue,
            step_loss_weights=tuple(args.trm_step_loss_weights) if args.trm_step_loss_weights is not None else None,
        ),
        brc=BRCConfig(
            recurrent_steps=args.brc_recurrent_steps,
            block_layers=args.brc_block_layers,
            num_heads=args.brc_num_heads,
            mlp_ratio=args.brc_mlp_ratio,
            position_encoding=args.brc_position_encoding,
            rms_norm_eps=args.brc_rms_norm_eps,
            rope_theta=args.brc_rope_theta,
            step_loss_weights=tuple(args.brc_step_loss_weights) if args.brc_step_loss_weights is not None else None,
            denoise_initial_prob=args.brc_denoise_initial_prob,
            denoise_trajectory_prob=args.brc_denoise_trajectory_prob,
            denoise_teacher_reveal_prob=args.brc_denoise_teacher_reveal_prob,
            denoise_mode_weights=(
                tuple(args.brc_denoise_mode_weights)
                if args.brc_denoise_mode_weights is not None
                else BRCConfig().denoise_mode_weights
            ),
            fixed_point_entropy_weight=args.brc_fixed_point_entropy_weight,
            fixed_point_loss_weight=args.brc_fixed_point_loss_weight,
            context_weight_reg_weight=args.brc_context_weight_reg_weight,
        ),
        urm=URMConfig(
            recurrent_steps=args.urm_recurrent_steps,
            deep_recursion=args.urm_deep_recursion,
            latent_recursion=args.urm_latent_recursion,
            block_layers=args.urm_block_layers,
            num_heads=args.urm_num_heads,
            mlp_ratio=args.urm_mlp_ratio,
            conv_kernel=args.urm_conv_kernel,
            puzzle_embed_ndim=args.urm_puzzle_embed_ndim,
            puzzle_embed_len=args.urm_puzzle_embed_len,
            rms_norm_eps=args.urm_rms_norm_eps,
            rope_theta=args.urm_rope_theta,
            halt_exploration_prob=args.urm_halt_exploration_prob,
            step_loss_weights=tuple(args.urm_step_loss_weights) if args.urm_step_loss_weights is not None else None,
        ),
    )
    optimizer = OptimizerConfig(
        optimizer_type=args.optimizer_type,
        learning_rate=args.learning_rate,
        puzzle_embed_learning_rate=args.puzzle_embed_learning_rate,
        lr_min_ratio=args.lr_min_ratio,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        puzzle_embed_weight_decay=args.puzzle_embed_weight_decay,
        puzzle_embed_coalesce_updates=args.puzzle_embed_coalesce_updates,
        lr_warmup_steps=args.lr_warmup_steps,
        grad_clip_norm=args.grad_clip_norm,
        flatten_optimizer=args.flatten_optimizer,
    )
    train = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        log_epochs=args.log_epochs,
        trm_train_mode=args.trm_train_mode,
        halt_loss_weight=args.halt_loss_weight,
        terminal_residual_weight=args.terminal_residual_weight,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        ema=EMAConfig(
            enabled=args.ema_enabled,
            decay=args.ema_decay,
        ),
    )
    eval_config = EvalConfig(
        batch_size=args.eval_batch_size,
        epochs=args.eval_epochs,
        diagnostics=args.eval_diagnostics,
        full_dataset=args.eval_full_dataset,
    )
    data = DataConfig(dataset_path=args.dataset_path)
    runtime = RuntimeConfig(
        compute_dtype=args.compute_dtype,
        data_parallel_devices=args.data_parallel_devices,
        prefetch_depth=args.prefetch_depth,
        prefetch_workers=args.prefetch_workers,
        train_dispatch_chunk=args.train_dispatch_chunk,
        profile_enabled=args.profile_enabled,
        profile_start_step=args.profile_start_step,
        profile_steps=args.profile_steps,
        profile_dir=args.profile_dir,
    )
    wandb = WandbConfig(
        enabled=args.wandb_enabled,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
    )
    return ExperimentConfig(
        task=task,
        model=model,
        optimizer=optimizer,
        train=train,
        eval=eval_config,
        data=data,
        runtime=runtime,
        wandb=wandb,
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
        num_puzzle_identifiers=dataset.spec.num_puzzle_identifiers,
        seq_len=dataset.spec.seq_len,
    )
    config = apply_epoch_budget(config, dataset)
    run_training(config, dataset, config_path=args.config, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
