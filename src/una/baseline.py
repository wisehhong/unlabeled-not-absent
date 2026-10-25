"""Plain supervised YOLO baseline for an end-to-end Kaggle submission.

This deliberately ignores the positive-unlabeled nature of the task: unlabeled
organisms are treated as background. It exists to exercise the full flow
(export -> train -> predict -> submission CSV) and to give a reference score.

Ultralytics is imported lazily so the rest of ``una`` works without it.
"""

from __future__ import annotations

import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import coco, paths

SUBMISSION_COLUMNS = (
    "annotation_id", "image_id", "category_id",
    "bbox_x", "bbox_y", "bbox_width", "bbox_height", "score",
)


def yolo_dir() -> Path:
    return paths.data_root() / "yolo"


def split_by_dive(dataset: coco.Dataset, val_fraction: float, seed: int) -> set[str]:
    """Return image uids for validation, choosing whole dives only.

    Dives are visited in a seeded random order and added while they still fit
    under the target, so one oversized dive cannot swallow the validation set.
    """
    dives = dataset.dives()
    order = sorted(dives)
    random.Random(seed).shuffle(order)

    target = val_fraction * len(dataset.images)
    val: set[str] = set()
    for dive in order:
        size = len(dives[dive])
        if len(val) + size <= target:
            val.update(img.uid for img in dives[dive])
    return val


def yolo_label_lines(
    image: coco.ImageRecord,
    annotations: list[coco.AnnotationRecord],
    categories: coco.CategoryMap,
) -> list[str]:
    """Return YOLO ``cls cx cy w h`` lines normalized by the original image size.

    Normalized coordinates are resize invariant, so labels stay valid for the
    downscaled cache images.
    """
    lines = []
    for ann in annotations:
        x, y, w, h = ann.bbox
        x0 = min(max(x, 0.0), image.width)
        y0 = min(max(y, 0.0), image.height)
        x1 = min(max(x + w, 0.0), image.width)
        y1 = min(max(y + h, 0.0), image.height)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        cls = categories.to_index[ann.category_id]
        cx = (x0 + x1) / 2 / image.width
        cy = (y0 + y1) / 2 / image.height
        nw = (x1 - x0) / image.width
        nh = (y1 - y0) / image.height
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def _resize_to_jpeg(src: Path, dst: Path, max_side: int) -> None:
    from PIL import Image

    if dst.is_file():
        return
    with Image.open(src) as img:
        img = img.convert("RGB")
        scale = max_side / max(img.size)
        if scale < 1:
            img = img.resize(
                (round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS
            )
        tmp = dst.with_suffix(".tmp")
        img.save(tmp, "JPEG", quality=95)
        tmp.replace(dst)


def export_yolo(max_side: int = 1280, val_fraction: float = 0.15, seed: int = 0,
                workers: int = 8) -> Path:
    """Write a YOLO dataset (resized images, labels, data.yaml) and return data.yaml."""
    dataset = coco.load("train")
    categories = dataset.categories
    val_uids = split_by_dive(dataset, val_fraction, seed)
    src_dir = paths.images_dir("train")
    root = yolo_dir()

    jobs = []
    counts = {"train": 0, "val": 0}
    for img in dataset.images:
        part = "val" if img.uid in val_uids else "train"
        counts[part] += 1
        stem = Path(img.file_name).stem
        image_dst = root / "images" / part / f"{stem}.jpg"
        label_dst = root / "labels" / part / f"{stem}.txt"
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        lines = yolo_label_lines(img, dataset.annotations_for(img.uid), categories)
        label_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        source = src_dir / img.file_name
        if not source.is_file():
            raise FileNotFoundError(f"missing training image {source}; run `una doctor`")
        jobs.append((source, image_dst))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(lambda j: _resize_to_jpeg(*j, max_side), jobs), 1):
            if i % 500 == 0 or i == len(jobs):
                print(f"  resized {i}/{len(jobs)}", flush=True)

    names = categories.index_names()
    yaml_lines = [
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
        *(f"  {i}: {json.dumps(name)}" for i, name in enumerate(names)),
    ]
    data_yaml = root / "data.yaml"
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (root / "classes.json").write_text(json.dumps(categories.to_json(), indent=2), encoding="utf-8")
    (root / "split.json").write_text(
        json.dumps({"seed": seed, "val_fraction": val_fraction, "counts": counts,
                    "val_dives": sorted({dataset.image(u).dive for u in val_uids})}, indent=2),
        encoding="utf-8",
    )
    print(f"  train={counts['train']} val={counts['val']} -> {data_yaml}")
    return data_yaml


def train(model: str = "yolo11s.pt", epochs: int = 30, imgsz: int = 1024, batch: int = 4,
          name: str = "yolo11s-baseline", workers: int = 4) -> Path:
    """Fine-tune a YOLO model on the exported dataset and return best.pt."""
    from ultralytics import YOLO

    data_yaml = yolo_dir() / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"{data_yaml} not found; run `una yolo-export` first")
    results = YOLO(model).train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, workers=workers,
        project=str(paths.runs_dir()), name=name, exist_ok=True,
        flipud=0.5, cos_lr=True, plots=True,
    )
    return Path(results.save_dir) / "weights" / "best.pt"


def predict(weights: Path, out: Path, imgsz: int = 1024, conf: float = 0.001,
            max_det: int = 300) -> Path:
    """Run the detector over the test split and write raw predictions as JSON.

    A very low confidence threshold is intentional: mAP rewards recall across
    the whole precision-recall curve, and low-score boxes cost nothing.
    """
    from ultralytics import YOLO

    dataset = coco.load("test")
    images_dir = paths.images_dir("test")
    model = YOLO(str(weights))

    records = []
    for i, img in enumerate(dataset.images, 1):
        source = images_dir / img.file_name
        result = model.predict(str(source), imgsz=imgsz, conf=conf, max_det=max_det,
                               verbose=False)[0]
        boxes = result.boxes
        records.append({
            "image_id": img.coco_image_id,
            "file_name": img.file_name,
            "width": img.width,
            "height": img.height,
            "cls": boxes.cls.int().tolist(),
            "xyxy": [[round(v, 2) for v in b] for b in boxes.xyxy.tolist()],
            "score": [round(s, 5) for s in boxes.conf.tolist()],
        })
        if i % 100 == 0 or i == len(dataset.images):
            print(f"  predicted {i}/{len(dataset.images)}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"weights": str(weights), "imgsz": imgsz, "conf": conf,
                               "max_det": max_det, "images": records}), encoding="utf-8")
    return out


def submission_rows(predictions: dict, categories: coco.CategoryMap) -> list[list]:
    """Convert raw predictions to Kaggle rows with original ids and COCO xywh boxes."""
    rows = []
    for rec in predictions["images"]:
        width, height = rec["width"], rec["height"]
        for cls, (x0, y0, x1, y1), score in zip(rec["cls"], rec["xyxy"], rec["score"]):
            x0, x1 = min(max(x0, 0.0), width), min(max(x1, 0.0), width)
            y0, y1 = min(max(y0, 0.0), height), min(max(y1, 0.0), height)
            w, h = x1 - x0, y1 - y0
            if w <= 0 or h <= 0:
                continue
            rows.append([
                len(rows) + 1,
                rec["image_id"],
                categories.to_category_id[cls],
                round(x0, 2), round(y0, 2), round(w, 2), round(h, 2),
                min(max(float(score), 0.0), 1.0),
            ])
    return rows


def write_submission(predictions_path: Path, out: Path) -> dict:
    """Write the submission CSV and return sanity statistics."""
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    categories = coco.load("train").categories
    test = coco.load("test")

    valid_categories = set(test.categories.names) or set(categories.names)
    valid_images = {img.coco_image_id for img in test.images}
    rows = submission_rows(predictions, categories)

    bad_cat = {r[2] for r in rows} - valid_categories
    bad_img = {r[1] for r in rows} - valid_images
    if bad_cat or bad_img:
        raise ValueError(f"invalid category ids {sorted(bad_cat)} or image ids {sorted(bad_img)[:5]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUBMISSION_COLUMNS)
        writer.writerows(rows)

    return {
        "rows": len(rows),
        "images_with_predictions": len({r[1] for r in rows}),
        "test_images": len(valid_images),
        "path": str(out),
    }
