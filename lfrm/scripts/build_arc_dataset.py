from __future__ import annotations

import argparse

from lfrm.datasets import build_arc_dataset, ensure_arc_source_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an offline ARC-AGI dataset matching TRM/URM preprocessing.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--input-file-prefix",
        type=str,
        default=None,
        help="Prefix before _training_challenges.json / _training_solutions.json, e.g. kaggle/combined/arc-agi.",
    )
    parser.add_argument(
        "--download-source",
        choices=("none", "samsung-trm-hf"),
        default="samsung-trm-hf",
        help="Download missing ARC source JSON files before building. Respects HF_ENDPOINT for mirrors.",
    )
    parser.add_argument(
        "--download-cache-prefix",
        type=str,
        default="data/raw/arc-agi/arc-agi",
        help="Local prefix used when --input-file-prefix is omitted.",
    )
    parser.add_argument("--overwrite-downloads", action="store_true")
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
    input_file_prefix = args.input_file_prefix or args.download_cache_prefix
    if args.download_source != "none":
        ensure_arc_source_files(
            input_file_prefix=input_file_prefix,
            subsets=tuple(args.subsets),
            source=args.download_source,
            overwrite=args.overwrite_downloads,
        )
    build_arc_dataset(
        output_dir=args.output_dir,
        input_file_prefix=input_file_prefix,
        subsets=tuple(args.subsets),
        test_set_name=args.test_set_name,
        test_set_name2=args.test_set_name2,
        seed=args.seed,
        num_aug=args.num_aug,
        puzzle_identifiers_start=args.puzzle_identifiers_start,
    )


if __name__ == "__main__":
    main()
