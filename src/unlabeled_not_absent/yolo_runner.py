"""YOLO inference and incomplete-annotation candidate generation."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .visualization import create_yolo_review_artifacts


def box_iou_xywh(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Return intersection-over-union for two pixel-space xywh boxes."""
    ax1, ay1 = float(first["x"]), float(first["y"])
    bx1, by1 = float(second["x"]), float(second["y"])
    ax2 = ax1 + max(0.0, float(first["width"]))
    ay2 = ay1 + max(0.0, float(first["height"]))
    bx2 = bx1 + max(0.0, float(second["width"]))
    by2 = by1 + max(0.0, float(second["height"]))
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def classify_predictions(
    predictions: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    *,
    overlap_threshold: float,
) -> list[dict[str, Any]]:
    """Annotate predictions using class-agnostic overlap with existing boxes."""
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be between 0 and 1")
    classified = []
    for index, prediction in enumerate(predictions):
        overlaps = [(box_iou_xywh(prediction, box), box) for box in annotations]
        best_iou, best_box = max(overlaps, key=lambda item: item[0], default=(0.0, None))
        classified.append(
            {
                **prediction,
                "prediction_id": prediction.get("prediction_id") or f"prediction-{index + 1}",
                "max_annotation_iou": round(float(best_iou), 6),
                "matched_annotation_uuid": best_box.get("uuid") if best_box else None,
                "matched_annotation_concept": best_box.get("concept") if best_box else None,
                "status": (
                    "overlaps_existing_annotation"
                    if best_iou >= overlap_threshold
                    else "missing_annotation_candidate"
                ),
            }
        )
    return classified


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _clip_xyxy(xyxy: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, x1), float(width))
    y2 = min(max(y2, y1), float(height))
    return x1, y1, x2, y2


def prediction_records(result: Any, image_width: int, image_height: int) -> list[dict[str, Any]]:
    """Convert one Ultralytics result into JSON-safe pixel-space records."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    coordinates = _as_numpy(boxes.xyxy).reshape(-1, 4)
    confidences = _as_numpy(boxes.conf).reshape(-1)
    classes = _as_numpy(boxes.cls).reshape(-1)
    names = getattr(result, "names", {})
    records = []
    for index, (xyxy, confidence, class_value) in enumerate(zip(coordinates, confidences, classes)):
        class_id = int(class_value)
        label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        x1, y1, x2, y2 = _clip_xyxy(
            [float(value) for value in xyxy], image_width, image_height
        )
        if x2 <= x1 or y2 <= y1:
            continue
        records.append(
            {
                "prediction_id": f"prediction-{index + 1}",
                "class_id": class_id,
                "label": str(label),
                "confidence": round(float(confidence), 6),
                "x": round(x1, 3),
                "y": round(y1, 3),
                "width": round(x2 - x1, 3),
                "height": round(y2 - y1, 3),
            }
        )
    return records


def _load_yolo(model: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "YOLO is not installed. Run `python -m pip install -e .[yolo]`, then rerun detect."
        ) from exc
    return YOLO(model)


def _model_provenance(detector: Any, requested_checkpoint: str) -> dict[str, Any]:
    checkpoint_value = getattr(detector, "ckpt_path", None) or requested_checkpoint
    checkpoint = Path(str(checkpoint_value))
    provenance: dict[str, Any] = {
        "requested_checkpoint": requested_checkpoint,
        "resolved_checkpoint": str(checkpoint_value),
    }
    if checkpoint.is_file():
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        provenance["checkpoint_sha256"] = digest.hexdigest()
    return provenance


def run_yolo(
    manifest_path: Path,
    *,
    output_dir: Path,
    model: str = "yolo11n.pt",
    confidence_threshold: float = 0.25,
    nms_iou_threshold: float = 0.7,
    overlap_threshold: float = 0.1,
    device: str | None = None,
    max_images: int | None = None,
) -> Path:
    """Run YOLO and write reviewable missing-annotation candidates."""
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not 0 <= nms_iou_threshold <= 1:
        raise ValueError("nms_iou_threshold must be between 0 and 1")
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be between 0 and 1")
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be positive")

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_dir = manifest_path.parent
    detector = _load_yolo(model)
    image_results: list[dict[str, Any]] = []

    for record in manifest["images"][:max_images]:
        image_path = sample_dir / record["local_file"]
        with Image.open(image_path) as loaded:
            width, height = loaded.size
        predict_args: dict[str, Any] = {
            "source": str(image_path),
            "conf": confidence_threshold,
            "iou": nms_iou_threshold,
            "verbose": False,
        }
        if device:
            predict_args["device"] = device
        inference_results = detector.predict(**predict_args)
        raw_predictions = prediction_records(inference_results[0], width, height)
        predictions = classify_predictions(
            raw_predictions,
            record.get("bounding_boxes") or [],
            overlap_threshold=overlap_threshold,
        )
        image_results.append(
            {
                "image_uuid": record["uuid"],
                "local_file": record["local_file"],
                "image_width": width,
                "image_height": height,
                "prediction_count": len(predictions),
                "candidate_count": sum(
                    item["status"] == "missing_annotation_candidate" for item in predictions
                ),
                "predictions": predictions,
            }
        )

    try:
        import ultralytics

        ultralytics_version = ultralytics.__version__
    except (ImportError, AttributeError):
        ultralytics_version = "unknown"
    candidate_count = sum(item["candidate_count"] for item in image_results)
    prediction_count = sum(item["prediction_count"] for item in image_results)
    model_provenance = _model_provenance(detector, model)
    model_provenance["ultralytics_version"] = ultralytics_version
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "method": "YOLO detection with class-agnostic IoU comparison against existing FathomNet boxes",
        "model": model_provenance,
        "parameters": {
            "confidence_threshold": confidence_threshold,
            "nms_iou_threshold": nms_iou_threshold,
            "annotation_overlap_threshold": overlap_threshold,
            "device": device or "auto",
        },
        "summary": {
            "images_processed": len(image_results),
            "predictions": prediction_count,
            "missing_annotation_candidates": candidate_count,
        },
        "important_limitations": [
            "An unmatched detection is a review candidate, not proof of a missing annotation.",
            "Overlap matching is class-agnostic because YOLO and FathomNet label vocabularies may differ.",
            "A generic COCO checkpoint is only a pipeline smoke test; marine conclusions require a suitable fine-tuned checkpoint.",
        ],
        "images": image_results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    create_yolo_review_artifacts(manifest, payload, sample_dir, output_dir)
    return result_path
