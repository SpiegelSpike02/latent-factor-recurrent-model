from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

from lfrm.config import ExperimentConfig
from lfrm.runtime.checkpoints import wandb_run_id_path
from lfrm.runtime.schedules import config_to_dict


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


def resolve_profile_dir(config: ExperimentConfig, checkpoint_dir: Path) -> Path:
    profile_dir = Path(config.runtime.profile_dir)
    if profile_dir.is_absolute():
        return profile_dir
    return checkpoint_dir / profile_dir


def patch_wandb_tensorboard(wandb_run, profile_dir: Path) -> None:
    if wandb_run is None:
        return
    with suppress(Exception):
        import wandb

        wandb.tensorboard.patch(root_logdir=str(profile_dir), save=True)


def upload_wandb_profile(wandb_run, profile_dir: Path, *, step: int) -> None:
    if wandb_run is None or not profile_dir.exists():
        return
    with suppress(Exception):
        import wandb

        artifact = wandb.Artifact(
            name=f"{wandb_run.id}-jax-profile-step-{step}",
            type="jax-profile",
            metadata={"step": step, "profile_dir": str(profile_dir)},
        )
        artifact.add_dir(str(profile_dir))
        wandb_run.log_artifact(artifact)
    with suppress(Exception):
        wandb_run.save(str(profile_dir / "**" / "*"), base_path=str(profile_dir.parent), policy="now")


def finish_wandb(wandb_run: Any) -> None:
    if wandb_run is not None:
        wandb_run.finish()
