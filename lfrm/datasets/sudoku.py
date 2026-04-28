from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


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
