from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import numpy as np


ARC_MAX_GRID_SIZE = 30
ARC_AUGMENT_RETRIES_FACTOR = 5
PUZZLE_ID_SEPARATOR = "|||"
ARC_NETWORK_SOURCES = {
    "samsung-trm-hf": (
        "https://huggingface.co",
        "wtfmahe/Samsung-TRM",
        "kaggle/combined",
    ),
}


@dataclass(frozen=True)
class ARCPuzzle:
    puzzle_id: str
    examples: tuple[tuple[np.ndarray, np.ndarray], ...]


def _dihedral_transform(arr: np.ndarray, transform_id: int) -> np.ndarray:
    if transform_id == 0:
        return arr
    if transform_id == 1:
        return np.rot90(arr, k=1)
    if transform_id == 2:
        return np.rot90(arr, k=2)
    if transform_id == 3:
        return np.rot90(arr, k=3)
    if transform_id == 4:
        return np.fliplr(arr)
    if transform_id == 5:
        return np.flipud(arr)
    if transform_id == 6:
        return arr.T
    if transform_id == 7:
        return np.fliplr(np.rot90(arr, k=1))
    raise ValueError(f"Unsupported dihedral transform id: {transform_id}")


def _arc_grid_to_np(grid: list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError("ARC grid must be 2D")
    if arr.shape[0] > ARC_MAX_GRID_SIZE or arr.shape[1] > ARC_MAX_GRID_SIZE:
        raise ValueError(f"ARC grid shape {arr.shape} exceeds {ARC_MAX_GRID_SIZE}x{ARC_MAX_GRID_SIZE}")
    if not np.all((arr >= 0) & (arr <= 9)):
        raise ValueError("ARC grid values must be in [0, 9]")
    return arr.astype(np.uint8)


def _grid_hash(grid: np.ndarray) -> str:
    if grid.ndim != 2 or grid.dtype != np.uint8:
        raise ValueError("ARC hash expects a uint8 2D grid")
    buffer = [int(x).to_bytes(1, byteorder="big") for x in grid.shape]
    buffer.append(grid.tobytes())
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def _puzzle_hash(puzzle_by_dest: dict[tuple[str, str], ARCPuzzle]) -> str:
    hashes: list[str] = []
    for puzzle in puzzle_by_dest.values():
        for input_grid, output_grid in puzzle.examples:
            hashes.append(f"{_grid_hash(input_grid)}|{_grid_hash(output_grid)}")
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def _sample_augmentation(name: str, rng: np.random.Generator):
    transform_id = int(rng.integers(0, 8))
    color_map = np.concatenate(
        [np.arange(0, 1, dtype=np.uint8), rng.permutation(np.arange(1, 10, dtype=np.uint8))]
    )
    aug_name = f"{name}{PUZZLE_ID_SEPARATOR}t{transform_id}{PUZZLE_ID_SEPARATOR}{''.join(str(x) for x in color_map)}"

    def map_grid(grid: np.ndarray) -> np.ndarray:
        return _dihedral_transform(color_map[grid], transform_id).astype(np.uint8, copy=False)

    return aug_name, map_grid


def _encode_arc_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    *,
    do_translation: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if do_translation:
        pad_row = int(rng.integers(0, ARC_MAX_GRID_SIZE - max(input_grid.shape[0], output_grid.shape[0]) + 1))
        pad_col = int(rng.integers(0, ARC_MAX_GRID_SIZE - max(input_grid.shape[1], output_grid.shape[1]) + 1))
    else:
        pad_row = 0
        pad_col = 0

    encoded: list[np.ndarray] = []
    for grid in (input_grid, output_grid):
        nrow, ncol = grid.shape
        padded = np.pad(
            grid + 2,
            ((pad_row, ARC_MAX_GRID_SIZE - pad_row - nrow), (pad_col, ARC_MAX_GRID_SIZE - pad_col - ncol)),
            constant_values=0,
        )
        eos_row = pad_row + nrow
        eos_col = pad_col + ncol
        if eos_row < ARC_MAX_GRID_SIZE:
            padded[eos_row, pad_col:eos_col] = 1
        if eos_col < ARC_MAX_GRID_SIZE:
            padded[pad_row:eos_row, eos_col] = 1
        encoded.append(padded.reshape(-1).astype(np.int32, copy=False))
    return encoded[0], encoded[1]


def _resolve_arc_file(input_file_prefix: str, subset: str, kind: str) -> Path:
    prefix = Path(input_file_prefix)
    candidates = (
        Path(f"{input_file_prefix}_{subset}_{kind}.json"),
        Path(f"{input_file_prefix}_{subset}-{kind}.json"),
        prefix.parent / f"{prefix.name}_{subset}_{kind}.json",
        prefix.parent / f"{prefix.name}_{subset}-{kind}.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find ARC {kind} file for subset={subset!r}, prefix={input_file_prefix!r}")


def _arc_file_path(input_file_prefix: str, subset: str, kind: str) -> Path:
    return Path(f"{input_file_prefix}_{subset}_{kind}.json")


def ensure_arc_source_files(
    *,
    input_file_prefix: str,
    subsets: tuple[str, ...],
    source: str = "samsung-trm-hf",
    overwrite: bool = False,
) -> None:
    if source not in ARC_NETWORK_SOURCES:
        raise ValueError(f"Unsupported ARC download source: {source!r}")
    default_endpoint, repo_id, repo_dir = ARC_NETWORK_SOURCES[source]
    endpoint = os.environ.get("HF_ENDPOINT", default_endpoint).rstrip("/")
    base_url = f"{endpoint}/{repo_id}/resolve/main/{repo_dir}"
    for subset in subsets:
        for kind in ("challenges", "solutions"):
            output_path = _arc_file_path(input_file_prefix, subset, kind)
            if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            filename = f"arc-agi_{subset}_{kind}.json"
            repo_filename = f"{repo_dir}/{filename}"
            url = f"{base_url}/{filename}"
            print(f"[arc] downloading {repo_id}/{repo_filename} -> {output_path}", flush=True)
            try:
                try:
                    from huggingface_hub import hf_hub_download

                    downloaded = hf_hub_download(
                        repo_id=repo_id,
                        filename=repo_filename,
                        repo_type="model",
                        endpoint=endpoint,
                    )
                    shutil.copyfile(downloaded, tmp_path)
                except Exception as hub_error:
                    print(f"[arc] huggingface_hub download failed ({hub_error}); falling back to {url}", flush=True)
                    urlretrieve(url, tmp_path)
                tmp_path.replace(output_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()


def _load_subset(input_file_prefix: str, subset: str) -> dict[str, dict]:
    challenges_path = _resolve_arc_file(input_file_prefix, subset, "challenges")
    puzzles = json.loads(challenges_path.read_text(encoding="utf-8"))
    try:
        solutions_path = _resolve_arc_file(input_file_prefix, subset, "solutions")
    except FileNotFoundError:
        print(f"{subset} solutions not found, filling with dummy")
        for puzzle in puzzles.values():
            for example in puzzle["test"]:
                example.setdefault("output", [[0]])
        return puzzles

    solutions = json.loads(solutions_path.read_text(encoding="utf-8"))
    for puzzle_id, solution_grids in solutions.items():
        for idx, solution_grid in enumerate(solution_grids):
            puzzles[puzzle_id]["test"][idx]["output"] = solution_grid
    return puzzles


def _convert_single_puzzle(
    results: dict[str, dict[str, list[list[ARCPuzzle]]]],
    name: str,
    puzzle: dict,
    *,
    aug_count: int,
    dest_mapping: dict[str, tuple[str, str]],
    rng: np.random.Generator,
) -> None:
    destinations = set(dest_mapping.values())
    converted = {dest: ARCPuzzle(name, ()) for dest in destinations}
    examples_by_dest: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {dest: [] for dest in destinations}
    for example_type, examples in puzzle.items():
        dest = dest_mapping[example_type]
        examples_by_dest[dest].extend(
            (_arc_grid_to_np(example["input"]), _arc_grid_to_np(example["output"]))
            for example in examples
        )
    converted = {dest: ARCPuzzle(name, tuple(examples)) for dest, examples in examples_by_dest.items()}

    group = [converted]
    if aug_count > 0:
        hashes = {_puzzle_hash(converted)}
        for _trial in range(ARC_AUGMENT_RETRIES_FACTOR * aug_count):
            aug_name, map_grid = _sample_augmentation(name, rng)
            augmented = {
                dest: ARCPuzzle(
                    aug_name,
                    tuple((map_grid(input_grid), map_grid(output_grid)) for input_grid, output_grid in arc_puzzle.examples),
                )
                for dest, arc_puzzle in converted.items()
            }
            puzzle_hash = _puzzle_hash(augmented)
            if puzzle_hash not in hashes:
                hashes.add(puzzle_hash)
                group.append(augmented)
            if len(group) >= aug_count + 1:
                break
        if len(group) < aug_count + 1:
            print(f"[Puzzle {name}] augmentation not full, only {len(group)}")

    for dest in destinations:
        split_name, set_name = dest
        results.setdefault(split_name, {}).setdefault(set_name, [])
        results[split_name][set_name].append([group_item[dest] for group_item in group])


def _load_arc_puzzles(
    *,
    input_file_prefix: str,
    subsets: Iterable[str],
    test_set_name: str,
    test_set_name2: str,
    num_aug: int,
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, list[list[ARCPuzzle]]]], dict[str, dict]]:
    train_dest = ("train", "all")
    test_map = {
        test_set_name: [(1.0, ("test", "all"))],
        test_set_name2: [(1.0, ("test", "all"))],
        "_default": [(1.0, ("train", "all"))],
    }
    test_puzzles: dict[str, dict] = {}
    results: dict[str, dict[str, list[list[ARCPuzzle]]]] = {}
    total_puzzles = 0
    for subset in subsets:
        puzzles = list(_load_subset(input_file_prefix, subset).items())
        rng.shuffle(puzzles)
        for idx, (name, puzzle) in enumerate(puzzles):
            fraction = idx / len(puzzles)
            test_dest = None
            for threshold, dest in test_map.get(subset, test_map["_default"]):
                if fraction < threshold:
                    test_dest = dest
                    break
            if test_dest is None:
                raise RuntimeError(f"Could not assign ARC subset {subset!r} to a destination")
            if test_dest[0] == "test":
                test_puzzles[name] = puzzle
            _convert_single_puzzle(
                results,
                name,
                puzzle,
                aug_count=num_aug,
                dest_mapping={"train": train_dest, "test": test_dest},
                rng=rng,
            )
            total_puzzles += 1
    print(f"Total puzzles: {total_puzzles}")
    return results, test_puzzles


def _write_split(
    output_root: Path,
    split_name: str,
    split: dict[str, list[list[ARCPuzzle]]],
    *,
    identifier_map: dict[str, int],
    num_identifiers: int,
    rng: np.random.Generator,
) -> None:
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    enable_translation = split_name == "train"

    total_examples = 0
    total_puzzles = 0
    total_groups = 0
    sets = list(split.keys())
    for set_name, groups in split.items():
        inputs: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        puzzle_identifiers: list[int] = []
        puzzle_indices = [0]
        group_indices = [0]
        example_id = 0
        puzzle_id = 0
        for group in groups:
            for puzzle in group:
                no_aug_id = int(rng.integers(0, len(puzzle.examples)))
                for example_idx, (input_grid, output_grid) in enumerate(puzzle.examples):
                    encoded_input, encoded_output = _encode_arc_pair(
                        input_grid,
                        output_grid,
                        do_translation=enable_translation and example_idx != no_aug_id,
                        rng=rng,
                    )
                    inputs.append(encoded_input)
                    labels.append(encoded_output)
                    example_id += 1
                    total_examples += 1
                puzzle_indices.append(example_id)
                puzzle_identifiers.append(identifier_map[puzzle.puzzle_id])
                puzzle_id += 1
                total_puzzles += 1
            group_indices.append(puzzle_id)
            total_groups += 1

        arrays = {
            "inputs": np.stack(inputs, axis=0).astype(np.int32, copy=False),
            "labels": np.stack(labels, axis=0).astype(np.int32, copy=False),
            "puzzle_identifiers": np.asarray(puzzle_identifiers, dtype=np.int32),
            "puzzle_indices": np.asarray(puzzle_indices, dtype=np.int32),
            "group_indices": np.asarray(group_indices, dtype=np.int32),
        }
        for key, value in arrays.items():
            np.save(split_dir / f"{set_name}__{key}.npy", value)
            if set_name == "all":
                np.save(split_dir / f"{key}.npy", value)

    metadata = {
        "kind": "arc",
        "task_type": "classification",
        "seq_len": ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE,
        "grid_height": ARC_MAX_GRID_SIZE,
        "grid_width": ARC_MAX_GRID_SIZE,
        "vocab_size": 12,
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": num_identifiers,
        "total_groups": total_groups,
        "mean_puzzle_examples": total_examples / max(total_puzzles, 1),
        "total_puzzles": total_puzzles,
        "num_examples": total_examples,
        "sets": sets,
    }
    (split_dir / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")


def build_arc_dataset(
    *,
    output_dir: str,
    input_file_prefix: str,
    subsets: tuple[str, ...],
    test_set_name: str,
    test_set_name2: str = "your_test_set",
    seed: int = 42,
    num_aug: int = 1000,
    puzzle_identifiers_start: int = 1,
) -> None:
    rng = np.random.default_rng(seed)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    data, test_puzzles = _load_arc_puzzles(
        input_file_prefix=input_file_prefix,
        subsets=subsets,
        test_set_name=test_set_name,
        test_set_name2=test_set_name2,
        num_aug=num_aug,
        rng=rng,
    )

    next_identifier = puzzle_identifiers_start
    identifier_map: dict[str, int] = {}
    for split in data.values():
        for subset in split.values():
            for group in subset:
                for puzzle in group:
                    if puzzle.puzzle_id not in identifier_map:
                        identifier_map[puzzle.puzzle_id] = next_identifier
                        next_identifier += 1
    print(f"Total puzzle IDs (including blank): {next_identifier}")

    for split_name, split in data.items():
        _write_split(
            output_root,
            split_name,
            split,
            identifier_map=identifier_map,
            num_identifiers=next_identifier,
            rng=rng,
        )

    ids_mapping = {value: key for key, value in identifier_map.items()}
    (output_root / "identifiers.json").write_text(
        json.dumps([ids_mapping.get(i, "") for i in range(next_identifier)]),
        encoding="utf-8",
    )
    (output_root / "test_puzzles.json").write_text(json.dumps(test_puzzles), encoding="utf-8")
