"""Command line entry point for the FathomNetCLEF2026 research pipeline."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

from . import baseline, coco, doctor, paths


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


def _cmd_yolo_export(args: argparse.Namespace) -> int:
    baseline.export_yolo(max_side=args.max_side, val_fraction=args.val_fraction,
                         seed=args.seed, workers=args.workers)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    best = baseline.train(model=args.model, epochs=args.epochs, imgsz=args.imgsz,
                          batch=args.batch, name=args.name, workers=args.workers)
    print(f"best weights: {best}")
    return 0


def _default_run_file(name: str, filename: str) -> Path:
    return paths.runs_dir() / name / filename


def _cmd_predict(args: argparse.Namespace) -> int:
    weights = args.weights or _default_run_file(args.name, "weights/best.pt")
    out = args.out or _default_run_file(args.name, "test_predictions.json")
    baseline.predict(weights, out, imgsz=args.imgsz, conf=args.conf, max_det=args.max_det)
    print(f"predictions: {out}")
    return 0


def _cmd_submission(args: argparse.Namespace) -> int:
    predictions = args.predictions or _default_run_file(args.name, "test_predictions.json")
    out = args.out or _default_run_file(args.name, "submission.csv")
    print(json.dumps(baseline.write_submission(predictions, out), indent=2))
    return 0


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

    default_name = "yolo11s-baseline"

    p_export = subparsers.add_parser(
        "yolo-export", help="write a resized YOLO dataset with a dive-grouped validation split")
    p_export.add_argument("--max-side", type=int, default=1280)
    p_export.add_argument("--val-fraction", type=float, default=0.15)
    p_export.add_argument("--seed", type=int, default=0)
    p_export.add_argument("--workers", type=int, default=8)
    p_export.set_defaults(func=_cmd_yolo_export)

    p_train = subparsers.add_parser("train", help="fine-tune the YOLO baseline")
    p_train.add_argument("--model", default="yolo11s.pt")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--imgsz", type=int, default=1024)
    p_train.add_argument("--batch", type=int, default=4)
    p_train.add_argument("--workers", type=int, default=4)
    p_train.add_argument("--name", default=default_name, help="run name under <data_root>/runs")
    p_train.set_defaults(func=_cmd_train)

    p_predict = subparsers.add_parser("predict", help="run the detector over the test split")
    p_predict.add_argument("--name", default=default_name)
    p_predict.add_argument("--weights", type=Path, help="default: runs/<name>/weights/best.pt")
    p_predict.add_argument("--out", type=Path, help="default: runs/<name>/test_predictions.json")
    p_predict.add_argument("--imgsz", type=int, default=1024)
    p_predict.add_argument("--conf", type=float, default=0.001)
    p_predict.add_argument("--max-det", type=int, default=300)
    p_predict.set_defaults(func=_cmd_predict)

    p_sub = subparsers.add_parser("submission", help="write the Kaggle submission CSV")
    p_sub.add_argument("--name", default=default_name)
    p_sub.add_argument("--predictions", type=Path,
                       help="default: runs/<name>/test_predictions.json")
    p_sub.add_argument("--out", type=Path, help="default: runs/<name>/submission.csv")
    p_sub.set_defaults(func=_cmd_submission)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
