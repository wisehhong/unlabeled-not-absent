"""Create reviewable FathomNet box and SAM mask visualizations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _color(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return tuple(70 + channel % 170 for channel in digest[:3])


def draw_boxes(image: Image.Image, boxes: list[dict[str, Any]]) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    scale = max(2, round(max(result.size) / 1000))
    for box in boxes:
        x, y = int(box["x"]), int(box["y"])
        x2, y2 = x + int(box["width"]), y + int(box["height"])
        label = str(box.get("concept") or "unknown")
        color = _color(label)
        draw.rectangle((x, y, x2, y2), outline=color, width=scale)
        text_box = draw.textbbox((x, y), label, font=font, stroke_width=1)
        draw.rectangle(text_box, fill=color)
        draw.text((x, y), label, fill="black", font=font, stroke_width=1, stroke_fill=color)
    return result


def overlay_masks(image: Image.Image, masks: list[tuple[str, np.ndarray]]) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    for label, mask in masks:
        mask = np.asarray(mask, dtype=bool)
        color = np.asarray(_color(label), dtype=np.float32)
        base[mask] = base[mask] * 0.55 + color * 0.45
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def create_review_artifacts(manifest: dict[str, Any], output_dir: Path) -> None:
    previews_dir = output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    thumbnails: list[tuple[Image.Image, str]] = []
    for record in manifest["images"]:
        with Image.open(output_dir / record["local_file"]) as image:
            preview = draw_boxes(image, record["bounding_boxes"])
            preview.thumbnail((640, 400))
            preview_path = previews_dir / f"{record['uuid']}.jpg"
            preview.save(preview_path, quality=90)
            evidence = ", ".join(record["habitat_review"]["heuristic_evidence"]) or "no benthic indicator"
            thumbnails.append((preview.copy(), f"{record['uuid'][:8]}  |  {evidence}"))

    if not thumbnails:
        return
    cell_w, cell_h = 660, 440
    columns = 2 if len(thumbnails) > 1 else 1
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (thumbnail, caption) in enumerate(thumbnails):
        left = (index % columns) * cell_w + 10
        top = (index // columns) * cell_h + 10
        sheet.paste(thumbnail, (left, top + 20))
        draw.text((left, top), caption, fill="black")
    sheet.save(output_dir / "contact_sheet.jpg", quality=92)

