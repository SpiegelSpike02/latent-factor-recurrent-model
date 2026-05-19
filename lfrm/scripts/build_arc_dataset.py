from __future__ import annotations

import argparse

from lfrm.datasets import build_arc_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an offline ARC-AGI dataset matching TRM/URM preprocessing.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--input-file-prefix",
        type=str,
        required=True,
        help="Prefix before _training_challenges.json / _training_solutions.json, e.g. kaggle/combined/arc-agi.",
    )
    parser.add_argument("--subsets", type=str, nargs="+", required=True)
    parser.add_argument("--test-set-name", type=str, required=True)
    parser.add_argument(
        "--test-set-name2",
        type=str,
        default="your_test_set",
        help="Optional second subset routed to test, matching TinyRecursiveModels' builder.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-aug", type=int, default=1000)
    parser.add_argument("--puzzle-identifiers-start", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_arc_dataset(
        output_dir=args.output_dir,
        input_file_prefix=args.input_file_prefix,
        subsets=tuple(args.subsets),
        test_set_name=args.test_set_name,
        test_set_name2=args.test_set_name2,
        seed=args.seed,
        num_aug=args.num_aug,
        puzzle_identifiers_start=args.puzzle_identifiers_start,
    )


if __name__ == "__main__":
    main()
