"""Selection and manifest helpers for a small benthic-first sample."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .fathomnet import API_URL, download_image, fetch_image_page, file_sha256, image_suffix

LOGGER = logging.getLogger(__name__)

# These labels are evidence that the frame is likely near the seafloor, not a
# ground-truth habitat classifier. Every selected frame remains reviewable.
BENTHIC_INDICATOR_TERMS = {
    "actiniarian",
    "anemone",
    "brittle star",
    "chiton",
    "coral",
    "crinoid",
    "echinoid",
    "sea cucumber",
    "sea fan",
    "sea pen",
    "sponge",
    "starfish",
    "substrate",
}


def benthic_evidence(image: dict[str, Any]) -> list[str]:
    concepts = {
        str(box.get("concept", "")).strip().lower()
        for box in image.get("boundingBoxes") or []
    }
    evidence = sorted(
        concept
        for concept in concepts
        if any(term in concept for term in BENTHIC_INDICATOR_TERMS)
    )
    return evidence


def is_usable(image: dict[str, Any]) -> bool:
    return bool(
        image.get("uuid")
        and image.get("url")
        and image.get("width")
        and image.get("height")
        and image.get("boundingBoxes")
        and image.get("valid", True)
    )


def collect_candidates(
    pages: Iterable[int],
    *,
    page_size: int,
    limit: int,
    habitat: str,
    api_url: str = API_URL,
) -> tuple[list[dict[str, Any]], list[int]]:
    pages = list(pages)
    selected: list[dict[str, Any]] = []
    fetched_pages: list[int] = []
    for page_index, page in enumerate(pages, start=1):
        LOGGER.info("Metadata page %d/%d: downloading API page %d", page_index, len(pages), page)
        response = fetch_image_page(page, page_size, api_url=api_url)
        fetched_pages.append(page)
        LOGGER.info(
            "Metadata page %d/%d: received %d records",
            page_index,
            len(pages),
            len(response["content"]),
        )
        for image in response["content"]:
            evidence = benthic_evidence(image)
            if not is_usable(image):
                continue
            if habitat == "benthic" and not evidence:
                continue
            selected.append({**image, "_benthicEvidence": evidence})
            if len(selected) >= limit:
                LOGGER.info("Metadata selection complete: selected %d/%d images", len(selected), limit)
                return selected, fetched_pages
        LOGGER.info("Metadata selection progress: selected %d/%d images", len(selected), limit)
    LOGGER.info("Metadata selection complete: selected %d/%d images", len(selected), limit)
    return selected, fetched_pages


def build_manifest(
    images: list[dict[str, Any]],
    *,
    pages: list[int],
    page_size: int,
    habitat: str,
) -> dict[str, Any]:
    records = []
    for image in images:
        filename = f"{image['uuid']}{image_suffix(image['url'])}"
        boxes = []
        for box in image.get("boundingBoxes") or []:
            boxes.append(
                {
                    key: box.get(key)
                    for key in (
                        "uuid",
                        "concept",
                        "x",
                        "y",
                        "width",
                        "height",
                        "reviewState",
                        "observer",
                    )
                }
            )
        records.append(
            {
                "uuid": image["uuid"],
                "source_url": image["url"],
                "sha256": image.get("sha256"),
                "local_file": f"images/{filename}",
                "width": image["width"],
                "height": image["height"],
                "depth_meters": image.get("depthMeters"),
                "timestamp": image.get("timestamp"),
                "bounding_boxes": boxes,
                "habitat_review": {
                    "status": "unreviewed_benthic_candidate" if habitat == "benthic" else "unreviewed",
                    "heuristic_evidence": image["_benthicEvidence"],
                    "reviewer_note": "",
                },
            }
        )
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "api": API_URL,
            "pages": pages,
            "page_size": page_size,
            "selection": habitat,
        },
        "important_limitations": [
            "Benthic selection is a concept-based candidate heuristic and requires visual confirmation.",
            "Existing boxes are prompts, not complete ground truth; absence of a box does not imply absence of an organism.",
            "Per project feedback, SAM 3 used labeled FathomNet data during training; report possible non-independence.",
        ],
        "images": records,
    }


def write_sample(manifest: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = manifest["images"]
    total = len(records)
    for index, record in enumerate(records, start=1):
        destination = output_dir / record["local_file"]
        expected_sha256 = record.get("sha256")
        if destination.exists() and expected_sha256:
            if file_sha256(destination) == str(expected_sha256).lower():
                LOGGER.info(
                    "Image %d/%d: reusing %s (SHA-256 matched)",
                    index,
                    total,
                    destination.name,
                )
                continue
            LOGGER.warning(
                "Image %d/%d: %s has a SHA-256 mismatch; downloading again",
                index,
                total,
                destination.name,
            )
        elif destination.exists():
            LOGGER.info("Image %d/%d: reusing %s (metadata has no SHA-256)", index, total, destination.name)
            continue
        else:
            LOGGER.info("Image %d/%d: downloading %s", index, total, destination.name)
        download_image(
            record["source_url"],
            destination,
            expected_sha256=str(expected_sha256) if expected_sha256 else None,
        )
        LOGGER.info("Image %d/%d: downloaded and verified %s", index, total, destination.name)
    manifest_path = output_dir / "manifest.json"
    LOGGER.info("Manifest: writing %s", manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
