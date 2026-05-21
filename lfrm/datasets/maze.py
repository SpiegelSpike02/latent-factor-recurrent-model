from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


MAZE_VOCAB = {
    "#": 1,
    " ": 2,
    "S": 3,
    "G": 4,
    "o": 5,
}


def _maybe_download_csv(source_repo: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download Maze CSVs from Hugging Face. "
            "Install it or provide --source-csv-dir."
        ) from exc
    return Path(hf_hub_download(source_repo, filename, repo_type="dataset"))


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


def _count_csv_rows(path: Path) -> int:
    with path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return sum(1 for _ in reader)


def _parse_grid(path: Path, value: str, *, grid_height: int, grid_width: int) -> np.ndarray:
    if len(value) != grid_height * grid_width:
        raise ValueError(
            f"Unexpected maze length in {path}: got {len(value)}, expected {grid_height * grid_width}"
        )
    try:
        encoded = np.fromiter((MAZE_VOCAB[ch] for ch in value), dtype=np.int32, count=len(value))
    except KeyError as exc:
        raise ValueError(f"Unexpected maze character {exc.args[0]!r} in {path}") from exc
    return encoded.reshape(grid_height, grid_width)


def _iter_csv_rows(path: Path, *, grid_height: int, grid_width: int):
    with path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            yield (
                _parse_grid(path, row["question"], grid_height=grid_height, grid_width=grid_width),
                _parse_grid(path, row["answer"], grid_height=grid_height, grid_width=grid_width),
            )


def _maze_transforms(grid: np.ndarray) -> tuple[np.ndarray, ...]:
    return (
        grid,
        np.rot90(grid, 1),
        np.rot90(grid, 2),
        np.rot90(grid, 3),
        np.fliplr(grid),
        np.flipud(grid),
        np.transpose(grid),
        np.fliplr(np.rot90(grid, 1)),
    )


def build_maze_dataset(
    *,
    output_dir: str,
    source_csv_dir: str | None,
    source_repo: str,
    subsample_size: int | None,
    aug: bool,
    seed: int,
    grid_height: int = 30,
    grid_width: int = 30,
    progress_every: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    def resolve_csv(filename: str) -> Path:
        if source_csv_dir is not None:
            path = Path(source_csv_dir) / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing Maze CSV: {path}")
            return path
        return _maybe_download_csv(source_repo, filename)

    split_map = {"train": "train.csv", "test": "test.csv"}
    csv_paths = {split_name: resolve_csv(filename) for split_name, filename in split_map.items()}
    source_counts: dict[str, int] = {}
    for split_name, csv_path in csv_paths.items():
        _progress(f"[{split_name}] counting rows in {csv_path}", progress_every)
        source_counts[split_name] = _count_csv_rows(csv_path)
        _progress(
            f"[{split_name}] found {_format_count(source_counts[split_name])} source examples",
            progress_every,
        )

    seq_len = grid_height * grid_width
    for split_name, csv_path in csv_paths.items():
        source_examples = source_counts[split_name]
        selected_indices: set[int] | None = None
        selected_examples = source_examples
        if split_name == "train" and subsample_size is not None and subsample_size < source_examples:
            selected = rng.choice(source_examples, size=subsample_size, replace=False)
            selected_indices = set(int(i) for i in selected)
            selected_examples = subsample_size
        num_variants = 8 if split_name == "train" and aug else 1
        num_examples = selected_examples * num_variants
        bytes_estimate = num_examples * (
            seq_len * np.dtype(np.uint8).itemsize * 2
            + np.dtype(np.int32).itemsize * 2
        ) + (selected_examples + 1) * np.dtype(np.int32).itemsize
        _progress(
            f"[{split_name}] writing {_format_count(num_examples)} examples "
            f"({selected_examples} sources x {num_variants} variants), "
            f"estimated files {_format_bytes(bytes_estimate)}",
            progress_every,
        )

        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        encoded_inputs = open_memmap(
            split_dir / "inputs.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(num_examples, seq_len),
        )
        encoded_labels = open_memmap(
            split_dir / "labels.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(num_examples, seq_len),
        )
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
        puzzle_identifiers = open_memmap(
            split_dir / "puzzle_identifiers.npy",
            mode="w+",
            dtype=np.int32,
            shape=(num_examples,),
        )

        output_idx = 0
        group_idx = 0
        puzzle_indices[0] = 0
        group_indices[0] = 0
        for source_idx, (maze, solution) in enumerate(
            _iter_csv_rows(csv_path, grid_height=grid_height, grid_width=grid_width)
        ):
            if selected_indices is not None and source_idx not in selected_indices:
                continue
            maze_variants = _maze_transforms(maze)
            solution_variants = _maze_transforms(solution)
            for aug_idx in range(num_variants):
                aug_maze = np.ascontiguousarray(maze_variants[aug_idx])
                aug_solution = np.ascontiguousarray(solution_variants[aug_idx])
                encoded_inputs[output_idx] = aug_maze.reshape(-1)
                encoded_labels[output_idx] = aug_solution.reshape(-1)
                puzzle_identifiers[output_idx] = 0
                output_idx += 1
                puzzle_indices[output_idx] = output_idx
                if progress_every > 0 and output_idx % progress_every == 0:
                    encoded_inputs.flush()
                    encoded_labels.flush()
                    puzzle_indices.flush()
                    group_indices.flush()
                    puzzle_identifiers.flush()
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
        puzzle_identifiers.flush()
        metadata = {
            "kind": "maze",
            "task_type": "classification",
            "seq_len": seq_len,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "vocab_size": 6,
            "num_puzzle_identifiers": 1,
            "num_examples": num_examples,
            "total_groups": selected_examples,
            "total_puzzles": num_examples,
            "mean_puzzle_examples": 1.0,
            "sets": ["all"],
            "pad_id": 0,
            "ignore_label_id": 0,
            "blank_identifier_id": MAZE_VOCAB[" "],
            "token_map": {"wall": 1, "blank": 2, "start": 3, "goal": 4, "path": 5},
        }
        (split_dir / "dataset.json").write_text(json.dumps(metadata))
        _progress(f"[{split_name}] finished {split_dir}", progress_every)
