from __future__ import annotations

import argparse

from data import build_sudoku_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an offline Sudoku dataset for JAX training.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--source-csv-dir", type=str, default=None, help="Optional local directory containing train.csv and test.csv.")
    parser.add_argument("--source-repo", type=str, default="sapientinc/sudoku-extreme")
    parser.add_argument("--subsample-size", type=int, default=None)
    parser.add_argument("--min-difficulty", type=int, default=None)
    parser.add_argument("--num-aug", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_sudoku_dataset(
        output_dir=args.output_dir,
        source_csv_dir=args.source_csv_dir,
        source_repo=args.source_repo,
        subsample_size=args.subsample_size,
        min_difficulty=args.min_difficulty,
        num_aug=args.num_aug,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
