from __future__ import annotations

from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from lfrm.config import ExperimentConfig
from lfrm.datasets import GridBatchSampler, dataset_overview
from lfrm.runtime.batching import BatchPrefetcher, sample_device_batch, small_metric_items
from lfrm.runtime.checkpoints import (
    build_run_checkpoint_dir,
    checkpoint_step,
    read_wandb_run_id,
    resolve_resume_checkpoint,
)
from lfrm.runtime.evaluation import evaluate
from lfrm.runtime.logging import (
    init_wandb,
    patch_wandb_tensorboard,
    resolve_profile_dir,
    upload_wandb_profile,
)
from lfrm.runtime.schedules import eval_interval_updates, eval_update_steps, schedule_learning_rate
from lfrm.runtime.sharding import (
    batch_sharding,
    data_parallel_mesh,
    place_module_replicated,
    place_tree,
    replicated_sharding,
)
from lfrm.training import (
    build_ema_update_runner,
    build_brc_step_carry_train_step_runner,
    build_state_copy_runner,
    build_eval_step_runner,
    build_train_step_runner,
    build_trm_act_train_step_runner,
    build_trm_dense_unroll_train_step_runner,
    build_trm_eval_step_runner,
    create_ema_model,
    create_model,
    create_optimizer,
    ema_param_filter,
    ema_sync_filter,
    load_checkpoint,
    save_checkpoint,
)
from lfrm.training.metrics import (
    WANDB_HISTORY_EXCLUDED_SCALAR_METRICS,
    flatten_step_metrics,
    format_step_summary,
    grouped_scalar_summary,
    optional_scalar_log,
    optional_summary_log,
    scalar_metric_names,
)


PER_STEP_SCALAR_SERIES = (
    ("per_step_loss", "loss_by_step"),
    ("per_step_accuracy", "accuracy_by_step"),
    ("per_step_q_top1_probability", "q_top1_probability_by_step"),
    ("per_step_update_step_size", "update_step_size_by_step"),
    ("per_step_update_rms", "update_rms_by_step"),
    ("per_step_velocity_rms", "velocity_rms_by_step"),
    ("per_step_energy_update_rms", "energy_update_rms_by_step"),
    ("per_step_energy_value", "energy_value_by_step"),
    ("per_step_energy_grad_rms", "energy_grad_rms_by_step"),
    ("per_step_logit_step_rms", "logit_step_rms_by_step"),
    ("per_step_distribution_tv_delta", "distribution_tv_delta_by_step"),
    ("per_step_path_energy", "path_energy_by_step"),
)


def validate_runtime_config(config: ExperimentConfig) -> None:
    if config.train.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.eval.batch_size < 0:
        raise ValueError("eval batch_size must be non-negative")
    if config.train.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.train.log_epochs <= 0:
        raise ValueError("log_epochs must be positive")
    if config.train.train_mode not in ("act", "dense_unroll", "step_carry"):
        raise ValueError("train_mode must be 'act', 'dense_unroll', or 'step_carry'")
    if config.eval.nums < 0:
        raise ValueError("eval nums must be non-negative; use 0 to disable eval")
    if not config.eval.full_dataset:
        raise ValueError("eval.full_dataset=false is not supported; eval is always full-dataset")
    if config.optimizer.lr_warmup_steps <= 0:
        raise ValueError("lr_warmup_steps must be positive")
    if not 0.0 <= config.optimizer.lr_min_ratio <= 1.0:
        raise ValueError("lr_min_ratio must be in [0, 1]")
    if config.optimizer.lr_mid_ratio < 0.0:
        raise ValueError("lr_mid_ratio must be non-negative")
    if config.optimizer.lr_mid_ratio > 0.0 and not 0.0 < config.optimizer.lr_mid_fraction < 1.0:
        raise ValueError("lr_mid_fraction must be in (0, 1) when lr_mid_ratio is enabled")
    if config.runtime.prefetch_depth <= 0:
        raise ValueError("prefetch_depth must be positive")
    if config.runtime.prefetch_workers <= 0:
        raise ValueError("prefetch_workers must be positive")
    if config.runtime.train_dispatch_chunk != 1:
        raise ValueError(
            "runtime.train_dispatch_chunk is reserved for a future scanned train runner; "
            "set it to 1 for the current optimized path"
        )
    if (
        config.runtime.data_parallel_devices != 1
        and config.optimizer.puzzle_embed_coalesce_updates
        and config.optimizer.puzzle_embed_learning_rate > 0.0
    ):
        print(
            "[runtime] sparse puzzle embedding updates are replicated before coalescing "
            "because jnp.unique/sort cannot run over a sharded data axis.",
            flush=True,
        )
    if config.runtime.profile_enabled:
        if config.runtime.profile_start_step <= 0:
            raise ValueError("profile_start_step must be positive when profiling is enabled")
        if config.runtime.profile_steps <= 0:
            raise ValueError("profile_steps must be positive when profiling is enabled")
    if config.model.model_type not in ("trm", "urm", "brc") and config.train.train_mode != "act":
        raise ValueError("train_mode is only supported for recurrent model types")
    if config.train.train_mode == "step_carry" and config.model.model_type != "brc":
        raise ValueError("train_mode='step_carry' is currently only implemented for model_type='brc'")
    if config.model.model_type == "brc" and config.train.train_mode != "step_carry":
        raise ValueError("BRC training supports only train_mode='step_carry'")


def training_uses_carry(config: ExperimentConfig) -> bool:
    return config.model.model_type in ("brc", "trm", "urm") and config.train.train_mode in ("act", "step_carry")


def halt_loss_weight_for_mode(config: ExperimentConfig) -> float:
    if config.model.model_type == "brc":
        return 0.0
    return config.train.halt_loss_weight


def validate_data_parallel_batching(config: ExperimentConfig, data_parallel_size: int) -> int:
    if config.train.batch_size % data_parallel_size != 0:
        raise ValueError(
            f"batch_size={config.train.batch_size} must be divisible by data_parallel_devices={data_parallel_size}"
        )
    eval_batch_size = config.eval.batch_size or config.train.batch_size
    if eval_batch_size % data_parallel_size != 0:
        raise ValueError(
            f"eval_batch_size={eval_batch_size} must be divisible by data_parallel_devices={data_parallel_size}"
        )
    return eval_batch_size


def build_step_runners(config: ExperimentConfig):
    halt_loss_weight = halt_loss_weight_for_mode(config)
    if config.model.model_type == "brc":
        train_step_fn = build_brc_step_carry_train_step_runner()
        eval_step_fn = build_eval_step_runner(
            halt_loss_weight,
            config.train.terminal_residual_weight,
        )
    elif config.model.model_type in ("trm", "urm"):
        if config.train.train_mode == "dense_unroll":
            train_step_fn = build_trm_dense_unroll_train_step_runner(
                halt_loss_weight=config.train.halt_loss_weight,
            )
        else:
            train_step_fn = build_trm_act_train_step_runner(config, config.train.halt_loss_weight)
        eval_step_fn = build_trm_eval_step_runner(
            config.train.halt_loss_weight,
            collect_diagnostics=config.eval.diagnostics,
        )
    else:
        train_step_fn = build_train_step_runner(
            config.train.halt_loss_weight,
            config.train.terminal_residual_weight,
        )
        eval_step_fn = build_eval_step_runner(
            config.train.halt_loss_weight,
            config.train.terminal_residual_weight,
        )
    return train_step_fn, eval_step_fn


def print_run_overview(
    *,
    config: ExperimentConfig,
    dataset,
    checkpoint_dir: Path,
    resume_checkpoint: Path | None,
    resume_step: int,
    data_parallel_size: int,
    data_sharding,
    profile_dir: Path,
) -> None:
    overview = dataset_overview(dataset)
    fields: list[tuple[str, object]] = [
        ("device", data_sharding or jax.devices()[0]),
        ("dataset_kind", overview["kind"]),
        ("task_type", overview["task_type"]),
        ("config_task_type", config.task.type),
        ("model_type", config.model.model_type),
        ("vocab_size", overview["vocab_size"]),
        ("input_vocab_size", overview["input_vocab_size"]),
    ]
    if config.model.model_type in ("trm", "urm"):
        fields.append(("num_puzzle_identifiers", overview["num_puzzle_identifiers"]))
    fields.extend(
        [
            ("train_examples", overview["train_examples"]),
            ("eval_examples", overview["eval_examples"]),
            ("batch_size", config.train.batch_size),
            ("eval_batch_size", config.eval.batch_size or config.train.batch_size),
            ("epochs", config.train.epochs),
            ("optimizer_updates", config.train.optimizer_updates),
            ("log_epochs", config.train.log_epochs),
            ("log_interval", config.train.log_interval_updates),
            ("eval_nums", config.eval.nums),
            ("eval_interval", eval_interval_updates(config)),
            ("lr_warmup_steps", config.optimizer.lr_warmup_steps),
            ("train_mode", config.train.train_mode),
            ("seq_len", config.model.seq_len),
            ("grid_height", config.model.grid_height),
            ("grid_width", config.model.grid_width),
            ("checkpoint_dir", checkpoint_dir),
            ("resume_checkpoint", resume_checkpoint),
            ("resume_step", resume_step),
            ("data_parallel_devices", data_parallel_size),
            ("data_sharding", data_sharding),
            ("prefetch_depth", config.runtime.prefetch_depth),
            ("prefetch_workers", config.runtime.prefetch_workers),
            ("train_dispatch_chunk", config.runtime.train_dispatch_chunk),
            ("eval_diagnostics", config.eval.diagnostics),
            ("profile_enabled", config.runtime.profile_enabled),
            ("profile_start_step", config.runtime.profile_start_step),
            ("profile_steps", config.runtime.profile_steps),
            ("profile_dir", profile_dir if config.runtime.profile_enabled else None),
        ]
    )
    print(" ".join(f"{name}= {value}" for name, value in fields))


def log_train_metrics(
    *,
    wandb_run,
    step: int,
    config: ExperimentConfig,
    host_metrics: dict[str, Any],
    scalar_metrics: tuple[str, ...],
    is_eval_step: bool,
    console_model_label: str,
) -> None:
    train_log = {
        "train/loss": float(host_metrics["loss"]),
        "train/learning_rate": schedule_learning_rate(config, step),
    }
    train_log.update(
        optional_scalar_log(
            "train",
            host_metrics,
            scalar_metrics,
            exclude_history=WANDB_HISTORY_EXCLUDED_SCALAR_METRICS,
        )
    )
    train_summary = optional_summary_log("train", host_metrics, WANDB_HISTORY_EXCLUDED_SCALAR_METRICS)
    if config.model.model_type != "brc":
        for metric_name, log_name in PER_STEP_SCALAR_SERIES:
            if metric_name in host_metrics:
                train_log.update(flatten_step_metrics(f"train/{log_name}", list(host_metrics[metric_name])))
    if wandb_run is not None:
        wandb_run.log(train_log, step=step, commit=not is_eval_step)
        for key, value in train_summary.items():
            wandb_run.summary[key] = value
    summary = grouped_scalar_summary(host_metrics, scalar_metrics, config.model.model_type)
    print(f"[train/{console_model_label}] step={step} lr={schedule_learning_rate(config, step):.2e}")
    if summary:
        print(summary)


def log_eval_metrics(
    *,
    wandb_run,
    step: int,
    config: ExperimentConfig,
    eval_metrics: dict[str, Any],
    scalar_metrics: tuple[str, ...],
    prefix: str,
    label: str,
    commit: bool,
    console_model_label: str,
) -> None:
    if wandb_run is not None:
        eval_log = {
            f"{prefix}/loss": eval_metrics["loss"],
        }
        eval_log.update(
            optional_scalar_log(
                prefix,
                eval_metrics,
                scalar_metrics,
                exclude_history=WANDB_HISTORY_EXCLUDED_SCALAR_METRICS,
            )
        )
        eval_summary = optional_summary_log(prefix, eval_metrics, WANDB_HISTORY_EXCLUDED_SCALAR_METRICS)
        for metric_name, log_name in PER_STEP_SCALAR_SERIES:
            if metric_name in eval_metrics:
                eval_log.update(flatten_step_metrics(f"{prefix}/{log_name}", eval_metrics[metric_name]))
        if "per_step_hidden_delta" in eval_metrics:
            eval_log.update(
                flatten_step_metrics(
                    f"{prefix}/hidden_delta_by_step",
                    eval_metrics["per_step_hidden_delta"],
                )
            )
        if "per_step_halt_probability" in eval_metrics:
            eval_log.update(
                flatten_step_metrics(
                    f"{prefix}/halt_probability_by_step",
                    eval_metrics["per_step_halt_probability"],
                )
            )
        wandb_run.log(eval_log, step=step, commit=commit)
        for key, value in eval_summary.items():
            wandb_run.summary[key] = value
    summary = grouped_scalar_summary(eval_metrics, scalar_metrics, config.model.model_type)
    print(f"[{label}/{console_model_label}] step={step}")
    if summary:
        print(summary)
    if "per_step_loss" in eval_metrics:
        print(
            "  "
            + " ".join(
                [
                    format_step_summary("loss", eval_metrics["per_step_loss"]),
                    format_step_summary("acc", eval_metrics.get("per_step_accuracy", [])),
                    format_step_summary("delta", eval_metrics.get("per_step_hidden_delta", [])),
                    format_step_summary("halt", eval_metrics.get("per_step_halt_probability", [])),
                ]
            )
        )


def run_training(
    config: ExperimentConfig,
    dataset,
    *,
    config_path: str | None,
    resume_from: str | None,
) -> None:
    validate_runtime_config(config)
    mesh = data_parallel_mesh(config)
    data_sharding = batch_sharding(mesh)
    state_sharding = replicated_sharding(mesh)
    data_parallel_size = 1 if mesh is None else int(mesh.shape["data"])
    validate_data_parallel_batching(config, data_parallel_size)

    resume_checkpoint: Path | None = None
    resume_step = 0
    resume_run_id: str | None = None
    if resume_from is not None:
        resume_checkpoint, checkpoint_dir = resolve_resume_checkpoint(resume_from)
        resume_step = checkpoint_step(resume_checkpoint) or 0
        resume_run_id = read_wandb_run_id(checkpoint_dir)
    else:
        checkpoint_dir = build_run_checkpoint_dir(config_path, config.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(config, run_dir=checkpoint_dir, resume_run_id=resume_run_id)
    profile_dir = resolve_profile_dir(config, checkpoint_dir)
    if config.runtime.profile_enabled:
        profile_dir.mkdir(parents=True, exist_ok=True)
        patch_wandb_tensorboard(wandb_run, profile_dir)

    model = create_model(config)
    optimizer = create_optimizer(model, config)
    train_step_fn, eval_step_fn = build_step_runners(config)
    ema_model = create_ema_model(model, config) if config.train.use_ema else None
    ema_update_fn = (
        build_ema_update_runner(config.train.ema_decay, ema_param_filter(config))
        if config.train.use_ema
        else None
    )
    ema_sync_fn = (
        build_state_copy_runner(ema_sync_filter(config))
        if config.train.use_ema
        else None
    )
    if resume_checkpoint is not None:
        restored_step = load_checkpoint(resume_checkpoint, model, optimizer, ema_model=ema_model)
        if restored_step != resume_step:
            resume_step = restored_step
    place_module_replicated(model, state_sharding)
    place_module_replicated(optimizer, state_sharding)
    if ema_model is not None:
        place_module_replicated(ema_model, state_sharding)

    train_rng = np.random.default_rng(config.train.seed + resume_step)
    train_sampler = GridBatchSampler(train_rng, dataset)
    train_key = jax.random.fold_in(jax.random.key(config.train.seed), resume_step)
    scalar_metrics = scalar_metric_names(config)

    print_run_overview(
        config=config,
        dataset=dataset,
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint=resume_checkpoint,
        resume_step=resume_step,
        data_parallel_size=data_parallel_size,
        data_sharding=data_sharding,
        profile_dir=profile_dir,
    )

    device = data_sharding or jax.devices()[0]

    def sample_train_batch():
        return sample_device_batch(
            train_sampler,
            dataset,
            config=config,
            split="train",
            device=device,
        )

    prefetcher = BatchPrefetcher(
        sample_train_batch,
        depth=config.runtime.prefetch_depth,
        workers=config.runtime.prefetch_workers,
    )

    current_batch = prefetcher.next()
    use_step_carry = training_uses_carry(config)
    console_model_label = "brc" if config.model.model_type == "brc" else config.model.model_type
    train_carry = place_tree(model.initial_carry(current_batch), data_sharding) if use_step_carry else None
    eval_steps = eval_update_steps(config)
    profile_active = False
    profile_finished = False
    profile_stop_step = config.runtime.profile_start_step + config.runtime.profile_steps - 1
    last_step = resume_step

    mesh_context = jax.sharding.set_mesh(mesh) if mesh is not None else nullcontext()
    try:
        with mesh_context:
            for step in range(resume_step + 1, config.train.optimizer_updates + 1):
                last_step = step
                if (
                    config.runtime.profile_enabled
                    and not profile_active
                    and not profile_finished
                    and step >= config.runtime.profile_start_step
                ):
                    print(f"[profile] start step={step} dir={profile_dir}", flush=True)
                    jax.profiler.start_trace(str(profile_dir))
                    profile_active = True
                is_eval_step = step in eval_steps
                if use_step_carry:
                    assert train_carry is not None
                    train_key, step_key = jax.random.split(train_key)
                    metrics, train_carry = train_step_fn(
                        model,
                        optimizer,
                        train_carry,
                        current_batch,
                        step_key,
                        jnp.asarray(step - 1, dtype=jnp.int32),
                    )
                else:
                    train_key, step_key = jax.random.split(train_key)
                    metrics = train_step_fn(
                        model,
                        optimizer,
                        current_batch,
                        step_key,
                    )
                current_batch = prefetcher.next()

                if ema_model is not None and ema_update_fn is not None:
                    ema_update_fn(ema_model, model)

                if step % config.train.log_interval_updates == 0 or step == 1:
                    host_metrics = jax.device_get(small_metric_items(metrics))
                    log_train_metrics(
                        wandb_run=wandb_run,
                        step=step,
                        config=config,
                        host_metrics=host_metrics,
                        scalar_metrics=scalar_metrics,
                        is_eval_step=is_eval_step,
                        console_model_label=console_model_label,
                    )

                if is_eval_step:
                    if ema_model is not None and ema_sync_fn is not None:
                        ema_sync_fn(ema_model, model)
                    save_checkpoint(str(checkpoint_dir), model, optimizer, step, ema_model=ema_model)

                    def run_eval_and_log(eval_model, prefix: str, label: str, *, commit: bool) -> dict[str, Any]:
                        eval_metrics = evaluate(
                            eval_step_fn,
                            eval_model,
                            dataset,
                            config=config,
                            device=device,
                        )
                        log_eval_metrics(
                            wandb_run=wandb_run,
                            step=step,
                            config=config,
                            eval_metrics=eval_metrics,
                            scalar_metrics=scalar_metrics,
                            prefix=prefix,
                            label=label,
                            commit=commit,
                            console_model_label=console_model_label,
                        )
                        return eval_metrics

                    if ema_model is not None:
                        run_eval_and_log(ema_model, "eval/ema", "eval/ema", commit=True)
                    else:
                        run_eval_and_log(model, "eval", "eval", commit=True)
                if profile_active and step >= profile_stop_step:
                    jax.profiler.stop_trace()
                    profile_active = False
                    profile_finished = True
                    print(f"[profile] stop step={step} dir={profile_dir}", flush=True)
                    upload_wandb_profile(wandb_run, profile_dir, step=step)
    finally:
        if profile_active:
            with suppress(Exception):
                jax.profiler.stop_trace()
            with suppress(Exception):
                upload_wandb_profile(wandb_run, profile_dir, step=last_step)
        with suppress(Exception):
            prefetcher.close()

    if wandb_run is not None:
        wandb_run.finish()
