from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lfrm.datasets import build_arc_dataset, load_dataset


class ARCDataTests(unittest.TestCase):
    def test_build_arc_dataset_matches_grid_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = root / "arc-agi"
            challenge = {
                "p0": {
                    "train": [{"input": [[1, 0], [0, 1]], "output": [[2, 2], [2, 2]]}],
                    "test": [{"input": [[0]], "output": [[1]]}],
                },
                "p1": {
                    "train": [{"input": [[3]], "output": [[4]]}],
                    "test": [{"input": [[5]], "output": [[6]]}],
                },
            }
            solution = {"p0": [[[1]]], "p1": [[[6]]]}
            for subset in ("training", "evaluation"):
                (root / f"arc-agi_{subset}_challenges.json").write_text(json.dumps(challenge), encoding="utf-8")
                (root / f"arc-agi_{subset}_solutions.json").write_text(json.dumps(solution), encoding="utf-8")

            output_dir = root / "arc1"
            build_arc_dataset(
                output_dir=str(output_dir),
                input_file_prefix=str(prefix),
                subsets=("training", "evaluation"),
                test_set_name="evaluation",
                num_aug=2,
                seed=42,
            )

            dataset = load_dataset(dataset_path=str(output_dir))
            self.assertEqual(dataset.spec.kind, "arc")
            self.assertEqual(dataset.spec.seq_len, 900)
            self.assertEqual(dataset.spec.grid_height, 30)
            self.assertEqual(dataset.spec.grid_width, 30)
            self.assertEqual(dataset.spec.vocab_size, 12)
            self.assertEqual(dataset.train_inputs.shape[1], 900)
            self.assertEqual(dataset.eval_inputs.shape[1], 900)
            self.assertTrue((output_dir / "train" / "all__inputs.npy").is_file())
            self.assertTrue((output_dir / "train" / "inputs.npy").is_file())


if __name__ == "__main__":
    unittest.main()
