from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfrm.datasets import DatasetSpec, GridBatchSampler, GridDataset, build_maze_dataset, load_dataset, sample_batch
from lfrm.datasets.maze import _maze_transforms
from lfrm.scripts.build_maze_dataset import build_parser


class MazeDatasetTests(unittest.TestCase):
    def test_build_and_load_maze_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_dir = root / "csv"
            csv_dir.mkdir()
            (csv_dir / "train.csv").write_text(
                "source,question,answer,rating\n"
                '0,"S  ##   G","Soo##  oG",1\n'
                '1,"S # #  G ","So#o#ooG ",1\n'
            )
            (csv_dir / "test.csv").write_text(
                "source,question,answer,rating\n"
                '0,"S  ##   G","Soo##  oG",1\n'
            )
            output_dir = root / "dataset"
            build_maze_dataset(
                output_dir=str(output_dir),
                source_csv_dir=str(csv_dir),
                source_repo="unused",
                subsample_size=None,
                aug=True,
                seed=0,
                grid_height=3,
                grid_width=3,
                progress_every=0,
            )

            dataset = load_dataset(dataset_path=str(output_dir))
            self.assertEqual(dataset.spec.kind, "maze")
            self.assertEqual(dataset.spec.vocab_size, 6)
            self.assertEqual(dataset.spec.seq_len, 9)
            self.assertEqual(dataset.train_inputs.shape, (16, 9))
            self.assertEqual(dataset.eval_inputs.shape, (1, 9))
            self.assertEqual(dataset.train_inputs.dtype.name, "uint8")
            self.assertEqual(dataset.train_labels.dtype.name, "uint8")
            self.assertEqual(dataset.train_puzzle_indices.tolist(), list(range(17)))
            self.assertEqual(dataset.train_group_indices.tolist(), [0, 8, 16])
            self.assertTrue(np.all(dataset.train_puzzle_identifiers == 0))
            self.assertTrue((output_dir / "train" / "inputs.npy").is_file())
            self.assertTrue((output_dir / "train" / "labels.npy").is_file())
            self.assertTrue((output_dir / "train" / "puzzle_identifiers.npy").is_file())
            self.assertTrue((output_dir / "train" / "puzzle_indices.npy").is_file())
            self.assertTrue((output_dir / "train" / "group_indices.npy").is_file())
            self.assertFalse((output_dir / "train" / "given_mask.npy").exists())

            batch = sample_batch(np.random.default_rng(0), dataset, batch_size=2, seq_len=9, split="train")
            self.assertEqual(batch["inputs"].shape, (2, 9))
            self.assertEqual(batch["labels"].shape, (2, 9))
            self.assertEqual(batch["puzzle_identifiers"].shape, (2,))

            sampler = GridBatchSampler(np.random.default_rng(0), dataset)
            grouped_batch = sampler.sample(batch_size=2, seq_len=9, split="train")
            selected_indices = []
            for row in grouped_batch["inputs"]:
                matches = np.where(np.all(dataset.train_inputs == row[None, :], axis=1))[0]
                self.assertGreater(matches.size, 0)
                selected_indices.append(int(matches[0]))
            selected_groups = {0 if index < 8 else 1 for index in selected_indices}
            self.assertEqual(selected_groups, {0, 1})

    def test_maze_transforms_match_official_dihedral_order(self) -> None:
        grid = np.arange(9).reshape(3, 3)
        transforms = _maze_transforms(grid)
        self.assertEqual(len(transforms), 8)
        np.testing.assert_array_equal(transforms[0], grid)
        np.testing.assert_array_equal(transforms[1], np.rot90(grid, 1))
        np.testing.assert_array_equal(transforms[2], np.rot90(grid, 2))
        np.testing.assert_array_equal(transforms[3], np.rot90(grid, 3))
        np.testing.assert_array_equal(transforms[4], np.fliplr(grid))
        np.testing.assert_array_equal(transforms[5], np.flipud(grid))
        np.testing.assert_array_equal(transforms[6], grid.T)
        np.testing.assert_array_equal(transforms[7], np.fliplr(np.rot90(grid, 1)))
        self.assertEqual({tuple(transform.reshape(-1)) for transform in transforms}, {
            tuple(transform.reshape(-1)) for transform in (
                grid,
                np.rot90(grid, 1),
                np.rot90(grid, 2),
                np.rot90(grid, 3),
                np.fliplr(grid),
                np.flipud(grid),
                grid.T,
                np.fliplr(np.rot90(grid, 1)),
            )
        })

    def test_grouped_sampler_packs_multiple_examples_from_selected_puzzle(self) -> None:
        inputs = np.asarray([[10, 11], [20, 21], [30, 31]], dtype=np.int32)
        labels = inputs + 100
        puzzle_identifiers = np.asarray([7, 7, 7], dtype=np.int32)
        dataset = GridDataset(
            train_inputs=inputs,
            train_labels=labels,
            eval_inputs=inputs[:1],
            eval_labels=labels[:1],
            train_puzzle_identifiers=puzzle_identifiers,
            eval_puzzle_identifiers=puzzle_identifiers[:1],
            train_puzzle_indices=np.asarray([0, 3], dtype=np.int32),
            eval_puzzle_indices=None,
            train_group_indices=np.asarray([0, 1], dtype=np.int32),
            eval_group_indices=None,
            spec=DatasetSpec(
                kind="arc",
                task_type="classification",
                vocab_size=128,
                input_vocab_size=128,
                num_puzzle_identifiers=8,
                total_groups=1,
                total_puzzles=1,
                mean_puzzle_examples=3.0,
                seq_len=2,
                grid_height=1,
                grid_width=2,
                train_examples=3,
                eval_examples=1,
            ),
        )
        batch = GridBatchSampler(np.random.default_rng(0), dataset).sample(
            batch_size=3,
            seq_len=2,
            split="train",
        )
        self.assertEqual({tuple(row) for row in batch["inputs"]}, {tuple(row) for row in inputs})
        np.testing.assert_array_equal(batch["puzzle_identifiers"], np.asarray([7, 7, 7], dtype=np.int32))

    def test_build_maze_parser_exposes_progress_every(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--output-dir", "dataset", "--progress-every", "7"])
        self.assertEqual(args.progress_every, 7)


if __name__ == "__main__":
    unittest.main()
