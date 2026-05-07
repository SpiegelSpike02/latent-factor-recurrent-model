from __future__ import annotations

import argparse

from lfrm.datasets import build_maze_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an offline Maze dataset for JAX training.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--source-csv-dir", type=str, default=None, help="Optional local directory containing train.csv and test.csv.")
    parser.add_argument("--source-repo", type=str, default="sapientinc/maze-30x30-hard-1k")
    parser.add_argument("--subsample-size", type=int, default=None)
    parser.add_argument("--aug", action="store_true", help="Match HRM preprocessing by writing all 8 train-time grid symmetries.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-height", type=int, default=30)
    parser.add_argument("--grid-width", type=int, default=30)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000,
        help="Print progress and flush mmap files every N written examples; use 0 to disable.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_maze_dataset(
        output_dir=args.output_dir,
        source_csv_dir=args.source_csv_dir,
        source_repo=args.source_repo,
        subsample_size=args.subsample_size,
        aug=args.aug,
        seed=args.seed,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
