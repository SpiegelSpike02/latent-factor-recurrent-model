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
    num_puzzle_identifiers: int
    total_groups: int
    total_puzzles: int
    mean_puzzle_examples: float
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
    train_puzzle_identifiers: np.ndarray
    eval_puzzle_identifiers: np.ndarray
    train_puzzle_indices: np.ndarray | None
    eval_puzzle_indices: np.ndarray | None
    train_group_indices: np.ndarray | None
    eval_group_indices: np.ndarray | None
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

    shared_keys = ("seq_len", "vocab_size", "num_puzzle_identifiers", "grid_height", "grid_width", "task_type")
    for key in shared_keys:
        if train_metadata[key] != eval_metadata[key]:
            raise ValueError(f"Train/eval metadata mismatch for '{key}'")

    train_inputs = _load_split_array(train_dir, "inputs")
    train_labels = _load_split_array(train_dir, "labels")
    eval_inputs = _load_split_array(eval_dir, "inputs")
    eval_labels = _load_split_array(eval_dir, "labels")
    train_given_mask = _load_given_mask(train_dir, train_inputs)
    eval_given_mask = _load_given_mask(eval_dir, eval_inputs)
    train_puzzle_indices = _load_optional_split_array(train_dir, "puzzle_indices")
    eval_puzzle_indices = _load_optional_split_array(eval_dir, "puzzle_indices")
    train_group_indices = _load_optional_split_array(train_dir, "group_indices")
    eval_group_indices = _load_optional_split_array(eval_dir, "group_indices")
    train_puzzle_identifiers = _load_puzzle_identifiers(train_dir, train_inputs.shape[0], train_puzzle_indices)
    eval_puzzle_identifiers = _load_puzzle_identifiers(eval_dir, eval_inputs.shape[0], eval_puzzle_indices)
    _validate_puzzle_identifiers(train_puzzle_identifiers, int(train_metadata["num_puzzle_identifiers"]), train_dir)
    _validate_puzzle_identifiers(eval_puzzle_identifiers, int(eval_metadata["num_puzzle_identifiers"]), eval_dir)
    _validate_indices(train_puzzle_indices, train_group_indices, train_inputs.shape[0], train_dir)
    _validate_indices(eval_puzzle_indices, eval_group_indices, eval_inputs.shape[0], eval_dir)

    spec = DatasetSpec(
        kind=str(train_metadata.get("kind", "grid")),
        task_type=str(train_metadata["task_type"]),
        vocab_size=int(train_metadata["vocab_size"]),
        num_puzzle_identifiers=int(train_metadata["num_puzzle_identifiers"]),
        total_groups=int(train_metadata["total_groups"]),
        total_puzzles=int(train_metadata["total_puzzles"]),
        mean_puzzle_examples=float(train_metadata["mean_puzzle_examples"]),
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
        train_puzzle_identifiers=train_puzzle_identifiers,
        eval_puzzle_identifiers=eval_puzzle_identifiers,
        train_puzzle_indices=train_puzzle_indices,
        eval_puzzle_indices=eval_puzzle_indices,
        train_group_indices=train_group_indices,
        eval_group_indices=eval_group_indices,
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
        puzzle_identifiers = dataset.train_puzzle_identifiers
        puzzle_indices = dataset.train_puzzle_indices
        group_indices = dataset.train_group_indices
    else:
        inputs = dataset.eval_inputs
        labels = dataset.eval_labels
        given_mask = dataset.eval_given_mask
        puzzle_identifiers = dataset.eval_puzzle_identifiers
        puzzle_indices = dataset.eval_puzzle_indices
        group_indices = dataset.eval_group_indices

    total = inputs.shape[0]
    if total == 0:
        raise ValueError(f"Split '{split}' is empty")
    if split == "train" and puzzle_indices is not None and group_indices is not None:
        indices = _sample_grouped_indices(rng, puzzle_indices, group_indices, batch_size)
    else:
        indices = rng.integers(total, size=batch_size, endpoint=False)

    batch_given_mask = np.asarray(given_mask[indices], dtype=bool)
    return {
        "inputs": np.asarray(inputs[indices], dtype=np.int32),
        "labels": np.asarray(labels[indices], dtype=np.int32),
        "given_mask": batch_given_mask,
        "puzzle_identifiers": np.asarray(puzzle_identifiers[indices], dtype=np.int32),
    }


def dataset_overview(dataset: GridDataset) -> dict[str, Any]:
    return {
        "kind": dataset.spec.kind,
        "vocab_size": dataset.spec.vocab_size,
        "num_puzzle_identifiers": dataset.spec.num_puzzle_identifiers,
        "total_groups": dataset.spec.total_groups,
        "total_puzzles": dataset.spec.total_puzzles,
        "mean_puzzle_examples": dataset.spec.mean_puzzle_examples,
        "train_examples": dataset.spec.train_examples,
        "eval_examples": dataset.spec.eval_examples,
        "seq_len": dataset.spec.seq_len,
        "grid_height": dataset.spec.grid_height,
        "grid_width": dataset.spec.grid_width,
        "task_type": dataset.spec.task_type,
    }


def _official_path(split_dir: Path, name: str) -> Path:
    return split_dir / f"all__{name}.npy"


def _plain_path(split_dir: Path, name: str) -> Path:
    return split_dir / f"{name}.npy"


def _split_path(split_dir: Path, name: str) -> Path:
    official = _official_path(split_dir, name)
    if official.is_file():
        return official
    return _plain_path(split_dir, name)


def _load_split_array(split_dir: Path, name: str) -> np.ndarray:
    path = _split_path(split_dir, name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return np.load(path, mmap_mode="r")


def _load_optional_split_array(split_dir: Path, name: str) -> np.ndarray | None:
    path = _split_path(split_dir, name)
    if not path.is_file():
        return None
    return np.load(path, mmap_mode=None)


def _load_given_mask(split_dir: Path, inputs: np.ndarray) -> np.ndarray:
    path = split_dir / "given_mask.npy"
    if not path.is_file():
        return np.asarray(inputs != 1, dtype=bool)
    return np.load(path, mmap_mode="r")


def _load_puzzle_identifiers(split_dir: Path, num_examples: int, puzzle_indices: np.ndarray | None) -> np.ndarray:
    path = _split_path(split_dir, "puzzle_identifiers")
    if not path.is_file():
        return np.zeros((num_examples,), dtype=np.int32)
    identifiers = np.load(path, mmap_mode="r")
    if identifiers.shape == (num_examples,):
        return identifiers
    if puzzle_indices is not None and identifiers.shape == (len(puzzle_indices) - 1,):
        return _expand_puzzle_identifiers(identifiers, puzzle_indices, num_examples)
    if identifiers.shape != (num_examples,):
        raise ValueError(
            f"Expected puzzle_identifiers shape {(num_examples,)} or {(len(puzzle_indices) - 1,) if puzzle_indices is not None else '(puzzles,)'}, "
            f"got {identifiers.shape} in {path}"
        )
    return identifiers


def _expand_puzzle_identifiers(identifiers: np.ndarray, puzzle_indices: np.ndarray, num_examples: int) -> np.ndarray:
    expanded = np.empty((num_examples,), dtype=np.int32)
    for puzzle_id, (start, end) in enumerate(zip(puzzle_indices[:-1], puzzle_indices[1:])):
        expanded[int(start) : int(end)] = int(identifiers[puzzle_id])
    return expanded


def _sample_grouped_indices(
    rng: np.random.Generator,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    group_ids = rng.integers(group_indices.size - 1, size=batch_size, endpoint=False)
    puzzle_lows = group_indices[group_ids]
    puzzle_highs = group_indices[group_ids + 1]
    puzzle_ids = rng.integers(puzzle_lows, puzzle_highs, endpoint=False)
    example_lows = puzzle_indices[puzzle_ids]
    example_highs = puzzle_indices[puzzle_ids + 1]
    return rng.integers(example_lows, example_highs, endpoint=False).astype(np.int64, copy=False)


def _validate_indices(
    puzzle_indices: np.ndarray | None,
    group_indices: np.ndarray | None,
    num_examples: int,
    split_dir: Path,
) -> None:
    if puzzle_indices is None and group_indices is None:
        return
    if puzzle_indices is None or group_indices is None:
        raise ValueError(f"{split_dir} must provide both puzzle_indices and group_indices")
    if puzzle_indices.ndim != 1 or group_indices.ndim != 1:
        raise ValueError(f"puzzle_indices and group_indices in {split_dir} must be 1D")
    if puzzle_indices.size < 2 or group_indices.size < 2:
        raise ValueError(f"puzzle_indices and group_indices in {split_dir} must contain boundaries")
    if int(puzzle_indices[0]) != 0 or int(puzzle_indices[-1]) != num_examples:
        raise ValueError(f"puzzle_indices in {split_dir} must start at 0 and end at {num_examples}")
    if int(group_indices[0]) != 0 or int(group_indices[-1]) != puzzle_indices.size - 1:
        raise ValueError(f"group_indices in {split_dir} must span all puzzles")
    if np.any(np.diff(puzzle_indices) < 0) or np.any(np.diff(group_indices) < 0):
        raise ValueError(f"indices in {split_dir} must be non-decreasing")


def _validate_puzzle_identifiers(identifiers: np.ndarray, num_puzzle_identifiers: int, split_dir: Path) -> None:
    if num_puzzle_identifiers < 1:
        raise ValueError(f"num_puzzle_identifiers must be at least 1 in {split_dir}")
    if identifiers.size == 0:
        return
    min_identifier = int(np.min(identifiers))
    max_identifier = int(np.max(identifiers))
    if min_identifier < 0 or max_identifier >= num_puzzle_identifiers:
        raise ValueError(
            f"puzzle_identifiers in {split_dir} must be in [0, {num_puzzle_identifiers}); "
            f"got min={min_identifier}, max={max_identifier}"
        )


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
    if "num_puzzle_identifiers" not in normalized:
        normalized["num_puzzle_identifiers"] = 1
    normalized.setdefault("total_groups", normalized.get("num_examples", 0))
    normalized.setdefault("total_puzzles", normalized.get("num_examples", 0))
    normalized.setdefault("mean_puzzle_examples", 1.0)
    return normalized
