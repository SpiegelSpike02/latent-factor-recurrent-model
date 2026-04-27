from __future__ import annotations

import csv
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


def _maybe_download_csv(source_repo: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download Sudoku CSVs from Hugging Face. "
            "Install it or provide --source-csv-dir."
        ) from exc
    downloaded = hf_hub_download(source_repo, filename, repo_type="dataset")
    return Path(downloaded)


def _load_csv_rows(path: Path, min_difficulty: int | None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    boards: list[np.ndarray] = []
    solutions: list[np.ndarray] = []
    with path.open(newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for _source, q, a, rating in reader:
            if (min_difficulty is None) or (int(rating) >= min_difficulty):
                if len(q) != 81 or len(a) != 81:
                    raise ValueError(f"Unexpected Sudoku lengths in {path}")
                boards.append(np.frombuffer(q.replace(".", "0").encode(), dtype=np.uint8).reshape(9, 9) - ord("0"))
                solutions.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord("0"))
    return boards, solutions


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray, *, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    digit_map = np.pad(rng.permutation(np.arange(1, 10)), (1, 0))
    transpose_flag = bool(rng.random() < 0.5)
    bands = rng.permutation(3)
    row_perm = np.concatenate([b * 3 + rng.permutation(3) for b in bands])
    stacks = rng.permutation(3)
    col_perm = np.concatenate([s * 3 + rng.permutation(3) for s in stacks])
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transform(x: np.ndarray) -> np.ndarray:
        if transpose_flag:
            x = x.T
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        return digit_map[new_board]

    return apply_transform(board), apply_transform(solution)


def _encode_sudoku_batch(seq: list[np.ndarray]) -> np.ndarray:
    arr = np.stack(seq, axis=0).reshape(len(seq), -1)
    if not np.all((arr >= 0) & (arr <= 9)):
        raise ValueError("Sudoku array contains values outside [0, 9]")
    return arr.astype(np.int32) + 1


def _given_mask_from_inputs(inputs: np.ndarray) -> np.ndarray:
    # Tokens are shifted by +1, so blank cells are exactly 1.
    return inputs != 1


def build_sudoku_dataset(
    *,
    output_dir: str,
    source_csv_dir: str | None,
    source_repo: str,
    subsample_size: int | None,
    min_difficulty: int | None,
    num_aug: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    def resolve_csv(filename: str) -> Path:
        if source_csv_dir is not None:
            path = Path(source_csv_dir) / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing Sudoku CSV: {path}")
            return path
        return _maybe_download_csv(source_repo, filename)

    split_map = {"train": "train.csv", "eval": "test.csv"}
    for split_name, filename in split_map.items():
        boards, solutions = _load_csv_rows(resolve_csv(filename), min_difficulty)
        if split_name == "train" and subsample_size is not None and subsample_size < len(boards):
            indices = rng.choice(len(boards), size=subsample_size, replace=False)
            boards = [boards[i] for i in indices]
            solutions = [solutions[i] for i in indices]

        num_augments = num_aug if split_name == "train" else 0
        all_inputs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for board, solution in zip(boards, solutions):
            for aug_idx in range(num_augments + 1):
                if aug_idx == 0:
                    aug_board, aug_solution = board, solution
                else:
                    aug_board, aug_solution = shuffle_sudoku(board, solution, rng=rng)
                all_inputs.append(aug_board)
                all_labels.append(aug_solution)

        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        encoded_inputs = _encode_sudoku_batch(all_inputs)
        encoded_labels = _encode_sudoku_batch(all_labels)
        np.save(split_dir / "inputs.npy", encoded_inputs)
        np.save(split_dir / "labels.npy", encoded_labels)
        np.save(split_dir / "given_mask.npy", _given_mask_from_inputs(encoded_inputs))
        metadata = {
            "kind": "sudoku",
            "task_type": "classification",
            "seq_len": 81,
            "grid_height": 9,
            "grid_width": 9,
            "vocab_size": 11,
            "num_examples": len(all_inputs),
            "pad_id": 0,
        }
        (split_dir / "dataset.json").write_text(json.dumps(metadata))
