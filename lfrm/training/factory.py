from __future__ import annotations

from dataclasses import replace

from flax import nnx

from lfrm.config import ExperimentConfig
from lfrm.models import BRCSudokuModel, TinyRecursiveModel, UnifiedReasoningModel
from lfrm.training.optim import build_optimizer, trainable_param_filter


GridReasoningModel = BRCSudokuModel | TinyRecursiveModel | UnifiedReasoningModel


def create_model(config: ExperimentConfig) -> GridReasoningModel:
    model_config = replace(config.model, task=config.task)
    if model_config.model_type == "trm":
        return TinyRecursiveModel(
            model_config,
            config.runtime,
            rngs=nnx.Rngs(config.train.seed),
        )
    if model_config.model_type == "brc_sudoku":
        return BRCSudokuModel(
            model_config,
            config.runtime,
            rngs=nnx.Rngs(config.train.seed),
        )
    if model_config.model_type == "urm":
        return UnifiedReasoningModel(
            model_config,
            config.runtime,
            rngs=nnx.Rngs(config.train.seed),
        )
    raise ValueError("Only model_type='trm', 'brc_sudoku', or 'urm' is supported")


def create_optimizer(model: GridReasoningModel, config: ExperimentConfig) -> nnx.Optimizer:
    return nnx.Optimizer(model, build_optimizer(config, model), wrt=trainable_param_filter(config))


def create_ema_model(model: GridReasoningModel, config: ExperimentConfig) -> GridReasoningModel:
    """Create an eval-only shadow model initialized from the current params."""
    ema_model = create_model(config)
    nnx.update(ema_model, nnx.state(model))
    return ema_model


def ema_param_filter(config: ExperimentConfig):
    del config
    return nnx.All(nnx.Param, nnx.Not(nnx.PathContains("puzzle_embed")))
