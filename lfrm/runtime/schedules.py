from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np

from lfrm.config import ExperimentConfig


def schedule_learning_rate(config: ExperimentConfig, step: int) -> float:
    optimizer_step = max(0, step - 1)
    optimizer_updates = max(1, config.train.optimizer_updates)
    warmup_steps = max(1, config.optimizer.lr_warmup_steps)
    peak = config.optimizer.learning_rate
    end = peak * config.optimizer.lr_min_ratio
    decay_updates = max(optimizer_updates, warmup_steps + 1)
    if optimizer_step <= warmup_steps:
        return peak * optimizer_step / max(warmup_steps, 1)
    progress = min(max((optimizer_step - warmup_steps) / max(decay_updates - warmup_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(end + (peak - end) * cosine)


def eval_interval_updates(config: ExperimentConfig) -> int:
    return max(1, config.eval.interval_updates)


def updates_from_epochs(dataset, batch_size: int, epochs: int) -> int:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return max(
        1,
        int(
            epochs
            * dataset.spec.total_groups
            * dataset.spec.mean_puzzle_examples
            / batch_size
        ),
    )


def apply_epoch_budget(config: ExperimentConfig, dataset) -> ExperimentConfig:
    train = replace(
        config.train,
        optimizer_updates=updates_from_epochs(dataset, config.train.batch_size, config.train.epochs),
        log_interval_updates=updates_from_epochs(dataset, config.train.batch_size, config.train.log_epochs),
    )
    eval_config = replace(
        config.eval,
        interval_updates=updates_from_epochs(dataset, config.train.batch_size, config.eval.epochs),
    )
    return ExperimentConfig(
        task=config.task,
        model=config.model,
        optimizer=config.optimizer,
        train=train,
        eval=eval_config,
        data=config.data,
        runtime=config.runtime,
        wandb=config.wandb,
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)
