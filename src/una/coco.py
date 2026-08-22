"""Loading and normalizing the FathomNetCLEF2026 COCO files.

The organizer files carry four structural traps that silently corrupt naive
pipelines. This module neutralizes all of them:

1. ``file_name`` is a FathomNet UUID and never matches the URL basename, so the
   UUID is the only reliable join key between the COCO record and a file on disk.
2. ``image_id`` values collide across splits (1,405 ids appear in both files), so
   ids are namespaced as ``"<split>:<id>"``.
3. ``category_id`` is not a class index. Ids span 1..41 with nine gaps, left over
   from a 41-class scheme that had classes deleted without renumbering. Training
   needs a dense contiguous mapping plus a reverse map for submission.
4. ``flickr_url`` is a byte-for-byte duplicate of ``coco_url`` and is ignored.

It also recovers provenance (institution and dive) from the URL, because a random
train/validation split leaks near-duplicate consecutive framegrabs and inflates
measured accuracy. One SOI dive alone accounts for 18% of the training images.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import paths

# Trailing "_PART02" style suffixes split a single dive across several directories.
# They must be stripped or the same dive lands on both sides of a split.
_PART_SUFFIX = re.compile(r"_PART\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class ImageRecord:
    """One image, with provenance resolved and ids namespaced."""

    uid: str
    """Namespaced identifier, ``"<split>:<coco_image_id>"``."""

    coco_image_id: int
    """Original id from the COCO file, required for submission rows."""

    split: str
    file_name: str
    """FathomNet UUID filename; also the name used on disk by ``download.py``."""

    url: str
    width: int
    height: int
    source: str
    """Institution: SOI, NOAA, HURL, MBARI or UNKNOWN."""

    dive: str
    """Stable dive identifier used for leakage-free grouped splitting."""

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class AnnotationRecord:
    """One bounding box in COCO ``[x, y, width, height]`` pixel convention."""

    annotation_id: int
    image_uid: str
    category_id: int
    bbox: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        return self.bbox[2] * self.bbox[3]


@dataclass
class CategoryMap:
    """Bidirectional map between sparse COCO category ids and dense indices."""

    names: dict[int, str]
    """COCO category id -> human readable name."""

    to_index: dict[int, int] = field(init=False)
    """COCO category id -> dense 0-based class index."""

    to_category_id: list[int] = field(init=False)
    """Dense class index -> COCO category id."""

    def __post_init__(self) -> None:
        ordered = sorted(self.names)
        self.to_index = {cid: i for i, cid in enumerate(ordered)}
        self.to_category_id = ordered

    @property
    def num_classes(self) -> int:
        return len(self.to_category_id)

    def index_names(self) -> list[str]:
        """Return class names ordered by dense index, for a YOLO ``data.yaml``."""
        return [self.names[cid] for cid in self.to_category_id]

    def to_json(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "index_to_category_id": self.to_category_id,
            "index_to_name": self.index_names(),
            "category_id_to_index": {str(k): v for k, v in self.to_index.items()},
        }


@dataclass
class Dataset:
    """A fully normalized split."""

    split: str
    images: list[ImageRecord]
    annotations: list[AnnotationRecord]
    categories: CategoryMap

    def __post_init__(self) -> None:
        self._by_uid = {img.uid: img for img in self.images}
        self._anns_by_uid: dict[str, list[AnnotationRecord]] = {img.uid: [] for img in self.images}
        for ann in self.annotations:
            self._anns_by_uid[ann.image_uid].append(ann)

    def image(self, uid: str) -> ImageRecord:
        return self._by_uid[uid]

    def annotations_for(self, uid: str) -> list[AnnotationRecord]:
        return self._anns_by_uid[uid]

    def dives(self) -> dict[str, list[ImageRecord]]:
        grouped: dict[str, list[ImageRecord]] = {}
        for img in self.images:
            grouped.setdefault(img.dive, []).append(img)
        return grouped

    def summary(self) -> dict:
        per_source: dict[str, int] = {}
        for img in self.images:
            per_source[img.source] = per_source.get(img.source, 0) + 1
        return {
            "split": self.split,
            "images": len(self.images),
            "annotations": len(self.annotations),
            "categories": self.categories.num_classes,
            "dives": len(self.dives()),
            "images_per_source": dict(sorted(per_source.items())),
        }


def parse_provenance(url: str) -> tuple[str, str]:
    """Return ``(source, dive)`` recovered from an image URL.

    Three distinct URL layouts appear across the two splits::

        cloudfront.net/full_images/SOI_FK190726_S0282_PART04/<file>
        hurlimage.soest.hawaii.edu/.../Localization_Scenes/D2-EX2301-07/<file>
        database.fathomnet.org/static/m3/framegrabs/Doc%20Ricketts/images/0536/<file>

    ``_PARTnn`` suffixes are stripped so that every part of one dive shares a dive
    id, and percent-encoded platform names such as ``Doc%20Ricketts`` are decoded.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    segments = [unquote(part) for part in parsed.path.split("/") if part]

    if "hurlimage" in host:
        dive = segments[-2] if len(segments) >= 2 else "UNKNOWN"
        return "HURL", f"HURL/{dive}"

    if "fathomnet.org" in host:
        # .../framegrabs/<platform>/images/<dive>/<file>
        if "framegrabs" in segments:
            start = segments.index("framegrabs")
            platform = segments[start + 1] if len(segments) > start + 1 else "UNKNOWN"
            dive = segments[start + 3] if len(segments) > start + 3 else "UNKNOWN"
            return "MBARI", f"MBARI/{platform}/{dive}"
        return "MBARI", "MBARI/UNKNOWN"

    if "full_images" in segments:
        start = segments.index("full_images")
        directory = segments[start + 1] if len(segments) > start + 1 else "UNKNOWN"
        dive = _PART_SUFFIX.sub("", directory)
        if dive.startswith("SOI_"):
            return "SOI", dive
        if dive.startswith("NOAA"):
            return "NOAA", dive
        return "UNKNOWN", dive

    return "UNKNOWN", "UNKNOWN"


def load(split: str, path: Path | None = None) -> Dataset:
    """Load and normalize one organizer COCO file.

    ``test_dataset.json`` legitimately contains zero annotations; it is an
    inference manifest whose ground truth is held by the Kaggle grader.
    """
    json_path = path or paths.coco_json(split)
    if not json_path.is_file():
        raise FileNotFoundError(f"COCO file not found: {json_path}")

    raw = json.loads(json_path.read_text(encoding="utf-8"))

    categories = CategoryMap(
        names={int(c["id"]): str(c["name"]) for c in raw.get("categories", [])}
    )

    images: list[ImageRecord] = []
    for item in raw.get("images", []):
        url = item["coco_url"]
        source, dive = parse_provenance(url)
        coco_image_id = int(item["id"])
        images.append(
            ImageRecord(
                uid=f"{split}:{coco_image_id}",
                coco_image_id=coco_image_id,
                split=split,
                file_name=item["file_name"],
                url=url,
                width=int(item["width"]),
                height=int(item["height"]),
                source=source,
                dive=dive,
            )
        )

    known = {img.uid for img in images}
    annotations: list[AnnotationRecord] = []
    for item in raw.get("annotations", []):
        image_uid = f"{split}:{int(item['image_id'])}"
        if image_uid not in known:
            raise ValueError(f"annotation {item['id']} references unknown image {image_uid}")
        x, y, w, h = (float(v) for v in item["bbox"])
        annotations.append(
            AnnotationRecord(
                annotation_id=int(item["id"]),
                image_uid=image_uid,
                category_id=int(item["category_id"]),
                bbox=(x, y, w, h),
            )
        )

    return Dataset(split=split, images=images, annotations=annotations, categories=categories)
