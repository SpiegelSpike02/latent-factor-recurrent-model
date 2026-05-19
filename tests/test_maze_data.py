from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfrm.datasets import build_maze_dataset, load_dataset, sample_batch
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
            self.assertEqual(dataset.train_puzzle_indices.tolist(), list(range(17)))
            self.assertEqual(dataset.train_group_indices.tolist(), [0, 8, 16])
            self.assertTrue(np.all(dataset.train_puzzle_identifiers == 0))
            self.assertTrue((output_dir / "train" / "inputs.npy").is_file())
            self.assertTrue((output_dir / "train" / "labels.npy").is_file())
            self.assertTrue((output_dir / "train" / "puzzle_identifiers.npy").is_file())
            self.assertTrue((output_dir / "train" / "puzzle_indices.npy").is_file())
            self.assertTrue((output_dir / "train" / "group_indices.npy").is_file())
            np.testing.assert_array_equal(dataset.train_given_mask[0], dataset.train_inputs[0] != 2)
            self.assertTrue(bool(np.any(dataset.train_given_mask)))
            self.assertTrue(bool(np.any(~dataset.train_given_mask)))

            batch = sample_batch(np.random.default_rng(0), dataset, batch_size=2, seq_len=9, split="train")
            self.assertEqual(batch["inputs"].shape, (2, 9))
            self.assertEqual(batch["labels"].shape, (2, 9))
            self.assertEqual(batch["given_mask"].shape, (2, 9))
            self.assertEqual(batch["puzzle_identifiers"].shape, (2,))

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

    def test_build_maze_parser_exposes_progress_every(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--output-dir", "dataset", "--progress-every", "7"])
        self.assertEqual(args.progress_every, 7)


if __name__ == "__main__":
    unittest.main()
