"""Command-line entry point for the proof of concept."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from .sam3_runner import run_sam3
from .sampling import build_manifest, collect_candidates, write_sample
from .visualization import create_review_artifacts
from .yolo_runner import run_yolo

LOGGER = logging.getLogger(__name__)


def _pages(value: str) -> list[int]:
    try:
        pages = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pages must be comma-separated integers") from exc
    if not pages or any(page < 0 for page in pages):
        raise argparse.ArgumentTypeError("pages must contain non-negative integers")
    return pages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="una-poc",
        description="Download FathomNet samples, run YOLO, and refine boxes with SAM 3.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="download images, metadata, boxes, and a contact sheet")
    fetch.add_argument("--pages", type=_pages, default=_pages("10,11,12"))
    fetch.add_argument("--page-size", type=int, default=10)
    fetch.add_argument("--limit", type=int, default=6)
    fetch.add_argument("--habitat", choices=("benthic", "any"), default="benthic")
    fetch.add_argument("--output", type=Path, default=Path("data/poc"))

    random_fetch = subparsers.add_parser("fetch-random", help="sample deterministic random API pages")
    random_fetch.add_argument("--seed", type=int, default=42)
    random_fetch.add_argument("--page-count", type=int, default=8)
    random_fetch.add_argument("--max-page", type=int, default=47000)
    random_fetch.add_argument("--page-size", type=int, default=10)
    random_fetch.add_argument("--limit", type=int, default=6)
    random_fetch.add_argument("--habitat", choices=("benthic", "any"), default="benthic")
    random_fetch.add_argument("--output", type=Path, default=Path("data/poc"))

    detect = subparsers.add_parser("detect", help="run YOLO and flag unmatched review candidates")
    detect.add_argument("--manifest", type=Path, default=Path("data/poc/manifest.json"))
    detect.add_argument("--output", type=Path, default=Path("data/poc/yolo"))
    detect.add_argument("--model", default="yolo11n.pt")
    detect.add_argument("--confidence", type=float, default=0.25)
    detect.add_argument("--nms-iou", type=float, default=0.7)
    detect.add_argument("--overlap-iou", type=float, default=0.1)
    detect.add_argument("--device")
    detect.add_argument("--max-images", type=int)

    segment = subparsers.add_parser("segment", help="run optional SAM 3 box-prompt inference")
    segment.add_argument("--manifest", type=Path, default=Path("data/poc/manifest.json"))
    segment.add_argument("--output", type=Path, default=Path("data/poc/sam3"))
    segment.add_argument("--confidence", type=float, default=0.3)
    segment.add_argument("--max-images", type=int)
    segment.add_argument("--max-boxes-per-image", type=int)
    segment.add_argument(
        "--detections", type=Path, help="YOLO results.json; prompts candidates instead of existing boxes"
    )
    segment.add_argument(
        "--include-overlaps",
        action="store_true",
        help="with --detections, prompt all detections instead of candidates only",
    )
    return parser


def _fetch(args: argparse.Namespace, pages: list[int]) -> int:
    LOGGER.info(
        "Fetch started: %d metadata pages, page size %d, image limit %d",
        len(pages),
        args.page_size,
        args.limit,
    )
    images, fetched_pages = collect_candidates(
        pages,
        page_size=args.page_size,
        limit=args.limit,
        habitat=args.habitat,
    )
    if not images:
        raise RuntimeError("No matching images found; try more pages or --habitat any")
    LOGGER.info("Manifest: building metadata for %d images", len(images))
    manifest = build_manifest(
        images,
        pages=fetched_pages,
        page_size=args.page_size,
        habitat=args.habitat,
    )
    manifest_path = write_sample(manifest, args.output)
    create_review_artifacts(manifest, args.output)
    LOGGER.info("Fetch complete: %d images written to %s", len(images), args.output)
    print(json.dumps({"images": len(images), "manifest": str(manifest_path), "pages": fetched_pages}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "fetch":
        return _fetch(args, args.pages)
    if args.command == "fetch-random":
        if args.page_count <= 0 or args.max_page <= 0:
            raise ValueError("page-count and max-page must be positive")
        rng = random.Random(args.seed)
        pages = rng.sample(range(args.max_page + 1), k=min(args.page_count, args.max_page + 1))
        return _fetch(args, pages)
    if args.command == "segment":
        result_path = run_sam3(
            args.manifest,
            output_dir=args.output,
            confidence_threshold=args.confidence,
            max_images=args.max_images,
            max_boxes_per_image=args.max_boxes_per_image,
            detections_path=args.detections,
            include_overlapping_detections=args.include_overlaps,
        )
        print(result_path)
        return 0
    if args.command == "detect":
        result_path = run_yolo(
            args.manifest,
            output_dir=args.output,
            model=args.model,
            confidence_threshold=args.confidence,
            nms_iou_threshold=args.nms_iou,
            overlap_threshold=args.overlap_iou,
            device=args.device,
            max_images=args.max_images,
        )
        print(result_path)
        return 0
    raise AssertionError("unreachable")
