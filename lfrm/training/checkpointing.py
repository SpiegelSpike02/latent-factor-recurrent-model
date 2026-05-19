from __future__ import annotations

from pathlib import Path

import jax
from flax import nnx
from flax.training import orbax_utils
from orbax.checkpoint import (
    Checkpointer,
    PyTreeCheckpointHandler,
    args as ocp_args,
)

from lfrm.training.factory import GridReasoningModel


def build_ema_update_runner(decay: float, wrt=nnx.Param):
    def update_ema_model(ema_model: GridReasoningModel, model: GridReasoningModel) -> None:
        ema_params = nnx.state(ema_model, wrt)
        model_params = nnx.state(model, wrt)
        nnx.update(ema_model, nnx.state(model))
        updated_params = jax.tree.map(
            lambda ema, current: decay * ema + (1.0 - decay) * current,
            ema_params,
            model_params,
        )
        nnx.update(ema_model, updated_params)

    return nnx.jit(update_ema_model)


def save_checkpoint(
    checkpoint_dir: str,
    model: GridReasoningModel,
    optimizer: nnx.Optimizer,
    step: int,
    *,
    ema_model: GridReasoningModel | None = None,
) -> None:
    checkpointer = Checkpointer(PyTreeCheckpointHandler())
    target_dir = Path(checkpoint_dir).resolve() / f"step_{step}"
    payload = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": step,
        "uses_ema_model": ema_model is not None,
    }
    if ema_model is not None:
        payload["ema_model"] = nnx.state(ema_model)
    checkpointer.save(target_dir, payload, force=True)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: GridReasoningModel,
    optimizer: nnx.Optimizer,
    *,
    ema_model: GridReasoningModel | None = None,
) -> int:
    checkpointer = Checkpointer(PyTreeCheckpointHandler())
    restore_target = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": 0,
        "uses_ema_model": False,
    }
    if ema_model is not None:
        restore_target["ema_model"] = nnx.state(ema_model)
    payload = checkpointer.restore(
        Path(checkpoint_path).resolve(),
        args=ocp_args.PyTreeRestore(
            item=restore_target,
            restore_args=orbax_utils.restore_args_from_target(restore_target),
            partial_restore=True,
        ),
    )
    nnx.update(model, payload["model"])
    nnx.update(optimizer, payload["optimizer"])
    if ema_model is not None:
        if "ema_model" in payload:
            nnx.update(ema_model, payload["ema_model"])
        else:
            nnx.update(ema_model, payload["model"])
    return int(payload["step"])
