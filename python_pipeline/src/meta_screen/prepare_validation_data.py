"""Command line entry point for preparing validation datasets."""

# Reading guide for R users:
# - This file is just a thin command-line wrapper around `validation_data.py`.
# - If you are used to one R script that mainly parses arguments and then calls
#   helper functions, this is that layer.

from __future__ import annotations

import argparse

from meta_screen.validation_data import download_archives, prepare_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and normalize ReviewCopilot validation datasets."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download source zip archives before preparing normalized CSVs.",
    )
    parser.add_argument(
        "--source-dir",
        default="data/source_archives",
        help="Directory containing the raw ReviewCopilot zip archives.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where normalized CSVs should be written.",
    )
    args = parser.parse_args()

    if args.download:
        download_archives(args.source_dir)

    summary = prepare_all(args.source_dir, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
