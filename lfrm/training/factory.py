from __future__ import annotations

from flax import nnx

from lfrm.config import ExperimentConfig
from lfrm.models import BRCSudokuModel, TinyRecursiveModel, UnifiedReasoningModel
from lfrm.training.optim import build_optimizer


GridReasoningModel = BRCSudokuModel | TinyRecursiveModel | UnifiedReasoningModel


def create_model(config: ExperimentConfig) -> GridReasoningModel:
    if config.model.model_type == "trm":
        return TinyRecursiveModel(
            config.model,
            config.runtime,
            rngs=nnx.Rngs(config.train.seed),
        )
    if config.model.model_type == "brc_sudoku":
        return BRCSudokuModel(
            config.model,
            config.runtime,
            rngs=nnx.Rngs(config.train.seed),
        )
    if config.model.model_type == "urm":
        return UnifiedReasoningModel(
            config.model,
            config.runtime,
            rngs=nnx.Rngs(config.train.seed),
        )
    raise ValueError("Only model_type='trm', 'brc_sudoku', or 'urm' is supported")


def create_optimizer(model: GridReasoningModel, config: ExperimentConfig) -> nnx.Optimizer:
    return nnx.Optimizer(model, build_optimizer(config, model), wrt=nnx.Param)


def create_ema_model(model: GridReasoningModel, config: ExperimentConfig) -> GridReasoningModel:
    """Create an eval-only shadow model initialized from the current params."""
    ema_model = create_model(config)
    nnx.update(ema_model, nnx.state(model))
    return ema_model


def ema_param_filter(config: ExperimentConfig):
    del config
    return nnx.All(nnx.Param, nnx.Not(nnx.PathContains("puzzle_embed")))
