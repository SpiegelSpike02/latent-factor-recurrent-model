from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


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


def _parse_sudoku_row(path: Path, q: str, a: str) -> tuple[np.ndarray, np.ndarray]:
    if len(q) != 81 or len(a) != 81:
        raise ValueError(f"Unexpected Sudoku lengths in {path}")
    if any(ch != "." and ch not in "123456789" for ch in q):
        raise ValueError(f"Sudoku puzzle in {path} must use '.' for blanks and digits 1-9")
    if any(ch not in "123456789" for ch in a):
        raise ValueError(f"Sudoku solution in {path} must contain only digits 1-9")
    board = np.fromiter((0 if ch == "." else int(ch) for ch in q), dtype=np.uint8, count=81).reshape(9, 9)
    solution = np.fromiter((int(ch) for ch in a), dtype=np.uint8, count=81).reshape(9, 9)
    return board, solution


def _iter_csv_rows(path: Path, min_difficulty: int | None):
    with path.open(newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for _source, q, a, rating in reader:
            if (min_difficulty is None) or (int(rating) >= min_difficulty):
                yield _parse_sudoku_row(path, q, a)


def _format_count(value: int) -> str:
    return f"{value:,}"


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _progress(message: str, progress_every: int) -> None:
    if progress_every > 0:
        print(message, flush=True)


def _count_csv_rows(path: Path, min_difficulty: int | None, *, progress_every: int) -> int:
    count = 0
    scanned = 0
    with path.open(newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for _source, _q, _a, rating in reader:
            scanned += 1
            if (min_difficulty is None) or (int(rating) >= min_difficulty):
                count += 1
            if progress_every > 0 and scanned % progress_every == 0:
                _progress(
                    f"  counted {_format_count(scanned)} rows from {path.name}; "
                    f"kept {_format_count(count)}",
                    progress_every,
                )
    return count


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


def _encode_sudoku_input(board: np.ndarray) -> np.ndarray:
    if not np.all((board >= 0) & (board <= 9)):
        raise ValueError("Sudoku array contains values outside [0, 9]")
    return board.reshape(-1).astype(np.uint8, copy=False)


def _encode_sudoku_label(solution: np.ndarray) -> np.ndarray:
    if not np.all((solution >= 1) & (solution <= 9)):
        raise ValueError("Sudoku solution contains values outside [1, 9]")
    return (solution.reshape(-1) - 1).astype(np.uint8, copy=False)


def build_sudoku_dataset(
    *,
    output_dir: str,
    source_csv_dir: str | None,
    source_repo: str,
    subsample_size: int | None,
    min_difficulty: int | None,
    num_aug: int,
    seed: int,
    progress_every: int = 0,
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

    split_map = {"train": "train.csv", "test": "test.csv"}
    csv_paths = {split_name: resolve_csv(filename) for split_name, filename in split_map.items()}
    source_counts: dict[str, int] = {}
    for split_name, csv_path in csv_paths.items():
        _progress(f"[{split_name}] counting rows in {csv_path}", progress_every)
        source_counts[split_name] = _count_csv_rows(
            csv_path,
            min_difficulty,
            progress_every=progress_every,
        )
        _progress(
            f"[{split_name}] kept {_format_count(source_counts[split_name])} source examples",
            progress_every,
        )

    for split_name, csv_path in csv_paths.items():
        source_examples = source_counts[split_name]
        selected_indices: set[int] | None = None
        selected_examples = source_examples
        if split_name == "train" and subsample_size is not None and subsample_size < source_examples:
            selected = rng.choice(source_examples, size=subsample_size, replace=False)
            selected_indices = set(int(i) for i in selected)
            selected_examples = subsample_size
        num_augments = num_aug if split_name == "train" else 0
        num_examples = selected_examples * (num_augments + 1)
        bytes_estimate = num_examples * (
            81 * np.dtype(np.uint8).itemsize * 2
            + np.dtype(np.int32).itemsize * 2
        ) + (selected_examples + 1) * np.dtype(np.int32).itemsize
        _progress(
            f"[{split_name}] writing {_format_count(num_examples)} examples "
            f"({selected_examples} sources x {num_augments + 1} variants), "
            f"estimated files {_format_bytes(bytes_estimate)}",
            progress_every,
        )

        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        encoded_inputs = open_memmap(split_dir / "inputs.npy", mode="w+", dtype=np.uint8, shape=(num_examples, 81))
        encoded_labels = open_memmap(split_dir / "labels.npy", mode="w+", dtype=np.uint8, shape=(num_examples, 81))
        puzzle_indices = open_memmap(
            split_dir / "puzzle_indices.npy",
            mode="w+",
            dtype=np.int32,
            shape=(num_examples + 1,),
        )
        group_indices = open_memmap(
            split_dir / "group_indices.npy",
            mode="w+",
            dtype=np.int32,
            shape=(selected_examples + 1,),
        )
        output_idx = 0
        group_idx = 0
        puzzle_indices[0] = 0
        group_indices[0] = 0
        for source_idx, (board, solution) in enumerate(_iter_csv_rows(csv_path, min_difficulty)):
            if selected_indices is not None and source_idx not in selected_indices:
                continue
            for aug_idx in range(num_augments + 1):
                if aug_idx == 0:
                    aug_board, aug_solution = board, solution
                else:
                    aug_board, aug_solution = shuffle_sudoku(board, solution, rng=rng)
                encoded_input = _encode_sudoku_input(aug_board)
                encoded_inputs[output_idx] = encoded_input
                encoded_labels[output_idx] = _encode_sudoku_label(aug_solution)
                output_idx += 1
                puzzle_indices[output_idx] = output_idx
                if progress_every > 0 and output_idx % progress_every == 0:
                    encoded_inputs.flush()
                    encoded_labels.flush()
                    puzzle_indices.flush()
                    group_indices.flush()
                    _progress(
                        f"[{split_name}] wrote {_format_count(output_idx)} / {_format_count(num_examples)} examples",
                        progress_every,
                    )
            group_idx += 1
            group_indices[group_idx] = output_idx
        if output_idx != num_examples:
            raise RuntimeError(f"Expected to write {num_examples} {split_name} examples, wrote {output_idx}")
        encoded_inputs.flush()
        encoded_labels.flush()
        puzzle_indices.flush()
        group_indices.flush()
        metadata = {
            "kind": "sudoku",
            "task_type": "classification",
            "seq_len": 81,
            "grid_height": 9,
            "grid_width": 9,
            "vocab_size": 9,
            "input_vocab_size": 10,
            "num_puzzle_identifiers": 1,
            "num_examples": num_examples,
            "total_groups": selected_examples,
            "total_puzzles": num_examples,
            "mean_puzzle_examples": 1.0,
            "sets": ["all"],
            "pad_id": None,
            "ignore_label_id": None,
            "blank_identifier_id": 0,
            "input_blank_id": 0,
            "label_digit_offset": 1,
        }
        (split_dir / "dataset.json").write_text(json.dumps(metadata))
        _progress(f"[{split_name}] finished {split_dir}", progress_every)
