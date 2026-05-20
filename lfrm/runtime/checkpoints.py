from __future__ import annotations

from datetime import datetime
from pathlib import Path


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
