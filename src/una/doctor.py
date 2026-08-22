"""Verify that locally downloaded images actually match the COCO manifest.

A partially completed download is the most dangerous failure mode in this project
because it is *biased*, not random: small HURL images (~0.85 MB) succeed while
large SOI 4K frames (~14 MB) time out. Since institution correlates with habitat,
camera and annotation style, a partial download quietly changes what the model is
trained on, and every downstream number looks plausible but is wrong.

``doctor`` resolves the manifest against the filesystem and writes an index that
downstream stages consume. Stages refuse to run unless ``coverage_complete`` is
true or the user explicitly opts into a subset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import coco, paths


@dataclass
class FileStatus:
    uid: str
    file_name: str
    present: bool
    size_bytes: int = 0
    problem: str = ""

    @property
    def ok(self) -> bool:
        return self.present and not self.problem


def _verify_dimensions(path: Path, expected: tuple[int, int]) -> str:
    """Return an empty string when the image opens and matches ``expected``."""
    try:
        from PIL import Image
    except ImportError:
        return ""

    try:
        with Image.open(path) as handle:
            handle.verify()
        with Image.open(path) as handle:
            actual = handle.size
    except Exception as exc:  # noqa: BLE001 - any decode failure means unusable
        return f"unreadable: {type(exc).__name__}"

    if actual != expected:
        return f"size mismatch: file {actual[0]}x{actual[1]} vs manifest {expected[0]}x{expected[1]}"
    return ""


def inspect(split: str, verify: bool = False) -> dict:
    """Resolve ``split`` against local disk and return an index dictionary."""
    dataset = coco.load(split)
    directory = paths.images_dir(split)

    statuses: list[FileStatus] = []
    for image in dataset.images:
        path = directory / image.file_name
        if not path.is_file():
            statuses.append(FileStatus(image.uid, image.file_name, present=False))
            continue

        size = path.stat().st_size
        problem = "empty file" if size == 0 else ""
        if not problem and verify:
            problem = _verify_dimensions(path, (image.width, image.height))
        statuses.append(
            FileStatus(image.uid, image.file_name, present=True, size_bytes=size, problem=problem)
        )

    by_uid = {status.uid: status for status in statuses}

    per_source: dict[str, dict[str, int]] = {}
    for image in dataset.images:
        bucket = per_source.setdefault(image.source, {"expected": 0, "present": 0, "bytes": 0})
        bucket["expected"] += 1
        status = by_uid[image.uid]
        if status.ok:
            bucket["present"] += 1
            bucket["bytes"] += status.size_bytes

    for bucket in per_source.values():
        bucket["missing"] = bucket["expected"] - bucket["present"]

    present = sum(1 for status in statuses if status.ok)
    problems = [
        {"uid": status.uid, "file_name": status.file_name, "problem": status.problem}
        for status in statuses
        if status.present and status.problem
    ]
    missing = [status.uid for status in statuses if not status.present]

    return {
        "split": split,
        "images_dir": str(directory),
        "verified": verify,
        "expected": len(dataset.images),
        "present": present,
        "missing": len(missing),
        "corrupt": len(problems),
        "coverage": round(present / len(dataset.images), 6) if dataset.images else 0.0,
        "coverage_complete": present == len(dataset.images),
        "total_bytes": sum(status.size_bytes for status in statuses if status.ok),
        "per_source": dict(sorted(per_source.items())),
        "missing_uids": missing,
        "problems": problems,
        "available": sorted(status.uid for status in statuses if status.ok),
    }


def write_index(split: str, verify: bool = False) -> tuple[Path, dict]:
    """Run :func:`inspect` and persist the result, returning ``(path, index)``."""
    index = inspect(split, verify=verify)
    destination = paths.index_path(split)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return destination, index


def load_index(split: str) -> dict:
    """Load a previously written index, raising a helpful error when absent."""
    path = paths.index_path(split)
    if not path.is_file():
        raise FileNotFoundError(f"no local index for split {split!r}; run `una doctor` first")
    return json.loads(path.read_text(encoding="utf-8"))
