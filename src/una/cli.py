"""Command line entry point for the FathomNetCLEF2026 research pipeline."""

from __future__ import annotations

import argparse
import json
import sys

from . import coco, doctor, paths


def _cmd_paths(args: argparse.Namespace) -> int:
    print(json.dumps(paths.describe(), indent=2))
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    for split in args.splits:
        dataset = coco.load(split)
        summary = dataset.summary()
        print(json.dumps(summary, indent=2))
        if args.categories and dataset.categories.num_classes:
            print(json.dumps(dataset.categories.to_json(), indent=2))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    exit_code = 0
    for split in args.splits:
        destination, index = doctor.write_index(split, verify=args.verify)

        print(f"[{split}] {index['present']}/{index['expected']} images present "
              f"({index['coverage'] * 100:.2f}%), {index['total_bytes'] / 1024 ** 3:.2f} GB")
        for source, stats in index["per_source"].items():
            print(f"    {source:<8} {stats['present']}/{stats['expected']} "
                  f"({stats['bytes'] / 1024 ** 3:.2f} GB, {stats['missing']} missing)")
        if index["corrupt"]:
            print(f"    {index['corrupt']} corrupt or mismatched file(s)")
            for problem in index["problems"][:10]:
                print(f"      {problem['file_name']}: {problem['problem']}")
        print(f"    index written to {destination}")

        if not index["coverage_complete"]:
            print(f"    INCOMPLETE: {index['missing']} missing. Re-run download.py to resume.",
                  file=sys.stderr)
            exit_code = 1
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="una",
        description="Positive-unlabeled marine organism detection on FathomNetCLEF2026.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_paths = subparsers.add_parser("paths", help="show resolved data locations")
    p_paths.set_defaults(func=_cmd_paths)

    p_summary = subparsers.add_parser("summary", help="summarize the organizer COCO files")
    p_summary.add_argument("--splits", nargs="+", default=list(paths.SPLITS), choices=paths.SPLITS)
    p_summary.add_argument("--categories", action="store_true", help="also print the class map")
    p_summary.set_defaults(func=_cmd_summary)

    p_doctor = subparsers.add_parser("doctor", help="check downloaded images against the manifest")
    p_doctor.add_argument("--splits", nargs="+", default=list(paths.SPLITS), choices=paths.SPLITS)
    p_doctor.add_argument(
        "--verify",
        action="store_true",
        help="decode every image and compare its dimensions with the manifest (slow but catches "
             "truncated downloads)",
    )
    p_doctor.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
