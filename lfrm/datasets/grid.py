from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    kind: str
    task_type: str
    vocab_size: int
    seq_len: int
    grid_height: int
    grid_width: int
    train_examples: int
    eval_examples: int


@dataclass(frozen=True)
class GridDataset:
    train_inputs: np.ndarray
    train_labels: np.ndarray
    eval_inputs: np.ndarray
    eval_labels: np.ndarray
    train_given_mask: np.ndarray
    eval_given_mask: np.ndarray
    spec: DatasetSpec


def load_dataset(*, dataset_path: str) -> GridDataset:
    root = Path(dataset_path)
    train_dir = root / "train"
    eval_dir = root / "eval"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train split: {train_dir}")
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Missing eval split: {eval_dir}")

    train_metadata = json.loads((train_dir / "dataset.json").read_text())
    eval_metadata = json.loads((eval_dir / "dataset.json").read_text())

    train_metadata = _normalize_grid_metadata(train_metadata)
    eval_metadata = _normalize_grid_metadata(eval_metadata)

    shared_keys = ("seq_len", "vocab_size", "grid_height", "grid_width", "task_type")
    for key in shared_keys:
        if train_metadata[key] != eval_metadata[key]:
            raise ValueError(f"Train/eval metadata mismatch for '{key}'")

    train_inputs = np.load(train_dir / "inputs.npy", mmap_mode="r")
    train_labels = np.load(train_dir / "labels.npy", mmap_mode="r")
    eval_inputs = np.load(eval_dir / "inputs.npy", mmap_mode="r")
    eval_labels = np.load(eval_dir / "labels.npy", mmap_mode="r")
    train_given_mask = _load_given_mask(train_dir)
    eval_given_mask = _load_given_mask(eval_dir)

    spec = DatasetSpec(
        kind=str(train_metadata.get("kind", "grid")),
        task_type=str(train_metadata["task_type"]),
        vocab_size=int(train_metadata["vocab_size"]),
        seq_len=int(train_metadata["seq_len"]),
        grid_height=int(train_metadata["grid_height"]),
        grid_width=int(train_metadata["grid_width"]),
        train_examples=int(train_inputs.shape[0]),
        eval_examples=int(eval_inputs.shape[0]),
    )
    return GridDataset(
        train_inputs=train_inputs,
        train_labels=train_labels,
        eval_inputs=eval_inputs,
        eval_labels=eval_labels,
        train_given_mask=train_given_mask,
        eval_given_mask=eval_given_mask,
        spec=spec,
    )


def sample_batch(
    rng: np.random.Generator,
    dataset: GridDataset,
    batch_size: int,
    seq_len: int,
    *,
    split: str,
) -> dict[str, np.ndarray]:
    if dataset.spec.seq_len != seq_len:
        raise ValueError(f"Requested seq_len={seq_len}, but dataset seq_len={dataset.spec.seq_len}")

    if split == "train":
        inputs = dataset.train_inputs
        labels = dataset.train_labels
        given_mask = dataset.train_given_mask
    else:
        inputs = dataset.eval_inputs
        labels = dataset.eval_labels
        given_mask = dataset.eval_given_mask

    total = inputs.shape[0]
    if total == 0:
        raise ValueError(f"Split '{split}' is empty")
    replace = batch_size > total
    indices = rng.choice(total, size=batch_size, replace=replace)

    batch_given_mask = np.asarray(given_mask[indices], dtype=bool)
    return {
        "inputs": np.asarray(inputs[indices], dtype=np.int32),
        "labels": np.asarray(labels[indices], dtype=np.int32),
        "given_mask": batch_given_mask,
    }


def dataset_overview(dataset: GridDataset) -> dict[str, Any]:
    return {
        "kind": dataset.spec.kind,
        "vocab_size": dataset.spec.vocab_size,
        "train_examples": dataset.spec.train_examples,
        "eval_examples": dataset.spec.eval_examples,
        "seq_len": dataset.spec.seq_len,
        "grid_height": dataset.spec.grid_height,
        "grid_width": dataset.spec.grid_width,
        "task_type": dataset.spec.task_type,
    }


def _load_given_mask(split_dir: Path) -> np.ndarray:
    path = split_dir / "given_mask.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing given mask: {path}")
    return np.load(path, mmap_mode="r")


def _normalize_grid_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    seq_len = int(normalized["seq_len"])
    if "grid_height" not in normalized or "grid_width" not in normalized:
        side = int(round(seq_len ** 0.5))
        if side * side != seq_len:
            raise ValueError("Missing grid metadata and seq_len is not a square number")
        normalized["grid_height"] = side
        normalized["grid_width"] = side
    normalized.setdefault("kind", "grid")
    return normalized
