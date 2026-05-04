from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfrm.datasets import build_sudoku_dataset, load_dataset, sample_batch
from lfrm.scripts.build_sudoku_dataset import build_parser


TRAIN_PUZZLE = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
TRAIN_SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
TEST_PUZZLE = "003020600900305001001806400008102900700000008006708200002609500800203009005010300"
TEST_SOLUTION = "483921657967345821251876493548132976729564138136798245372689514814253769695417382"


class SudokuDataTests(unittest.TestCase):
    def test_build_and_load_sudoku_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_dir = root / "csv"
            csv_dir.mkdir()
            self._write_csv(csv_dir / "train.csv", TRAIN_PUZZLE, TRAIN_SOLUTION)
            self._write_csv(csv_dir / "test.csv", TEST_PUZZLE, TEST_SOLUTION)

            output_dir = root / "dataset"
            build_sudoku_dataset(
                output_dir=str(output_dir),
                source_csv_dir=str(csv_dir),
                source_repo="unused",
                subsample_size=None,
                min_difficulty=None,
                num_aug=1,
                seed=0,
            )

            dataset = load_dataset(
                dataset_path=str(output_dir),
            )

            self.assertEqual(dataset.spec.kind, "sudoku")
            self.assertEqual(dataset.spec.vocab_size, 11)
            self.assertEqual(dataset.spec.num_puzzle_identifiers, 1)
            self.assertEqual(dataset.spec.total_groups, 1)
            self.assertEqual(dataset.spec.total_puzzles, 2)
            self.assertEqual(dataset.spec.mean_puzzle_examples, 1.0)
            self.assertEqual(dataset.spec.seq_len, 81)
            self.assertEqual(dataset.train_inputs.shape, (2, 81))
            self.assertEqual(dataset.eval_inputs.shape, (1, 81))
            self.assertEqual(dataset.train_puzzle_identifiers.shape, (2,))
            self.assertEqual(dataset.eval_puzzle_identifiers.shape, (1,))
            self.assertEqual(dataset.train_puzzle_indices.tolist(), [0, 1, 2])
            self.assertEqual(dataset.train_group_indices.tolist(), [0, 2])
            self.assertTrue(np.all(dataset.train_puzzle_identifiers == 0))
            self.assertTrue((output_dir / "train" / "all__inputs.npy").is_file())
            self.assertTrue((output_dir / "train" / "all__labels.npy").is_file())
            self.assertTrue((output_dir / "train" / "all__puzzle_identifiers.npy").is_file())
            self.assertTrue((output_dir / "train" / "all__puzzle_indices.npy").is_file())
            self.assertTrue((output_dir / "train" / "all__group_indices.npy").is_file())
            self.assertFalse((output_dir / "train" / "given_mask.npy").exists())

            rng = np.random.default_rng(0)
            batch = sample_batch(rng, dataset, batch_size=2, seq_len=81, split="train")
            self.assertEqual(batch["inputs"].shape, (2, 81))
            self.assertEqual(batch["labels"].shape, (2, 81))
            self.assertEqual(batch["given_mask"].dtype, np.bool_)
            self.assertEqual(batch["puzzle_identifiers"].shape, (2,))
            self.assertTrue(np.all(batch["puzzle_identifiers"] == 0))
            self.assertTrue(np.any(batch["given_mask"]))
            self.assertTrue(np.any(~batch["given_mask"]))

    def test_load_legacy_sudoku_dataset_without_puzzle_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_dir = root / "csv"
            csv_dir.mkdir()
            self._write_csv(csv_dir / "train.csv", TRAIN_PUZZLE, TRAIN_SOLUTION)
            self._write_csv(csv_dir / "test.csv", TEST_PUZZLE, TEST_SOLUTION)

            output_dir = root / "dataset"
            build_sudoku_dataset(
                output_dir=str(output_dir),
                source_csv_dir=str(csv_dir),
                source_repo="unused",
                subsample_size=None,
                min_difficulty=None,
                num_aug=1,
                seed=0,
            )
            for split in ("train", "eval"):
                split_dir = output_dir / split
                (split_dir / "all__puzzle_identifiers.npy").unlink()
                metadata_path = split_dir / "dataset.json"
                metadata = json.loads(metadata_path.read_text())
                metadata.pop("num_puzzle_identifiers")
                metadata_path.write_text(json.dumps(metadata))

            dataset = load_dataset(dataset_path=str(output_dir))

            self.assertEqual(dataset.spec.num_puzzle_identifiers, 1)
            self.assertTrue(np.all(dataset.train_puzzle_identifiers == 0))
            self.assertTrue(np.all(dataset.eval_puzzle_identifiers == 0))

    def test_build_sudoku_parser_exposes_progress_every(self) -> None:
        args = build_parser().parse_args(
            [
                "--output-dir",
                "dataset",
                "--progress-every",
                "123",
            ]
        )
        self.assertEqual(args.progress_every, 123)

    @staticmethod
    def _write_csv(path: Path, puzzle: str, solution: str) -> None:
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "q", "a", "rating"])
            writer.writerow(["local", puzzle, solution, "10"])


if __name__ == "__main__":
    unittest.main()
