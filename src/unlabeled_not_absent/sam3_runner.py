"""Optional SAM 3 box-prompt inference for an already downloaded sample."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .visualization import overlay_masks


def normalize_xywh(box: dict[str, Any], image_width: int, image_height: int) -> list[float]:
    """Convert pixel xywh into the normalized cxcywh format required by SAM 3."""
    x = float(box["x"])
    y = float(box["y"])
    width = float(box["width"])
    height = float(box["height"])
    if image_width <= 0 or image_height <= 0 or width <= 0 or height <= 0:
        raise ValueError("image and box dimensions must be positive")
    return [
        (x + width / 2) / image_width,
        (y + height / 2) / image_height,
        width / image_width,
        height / image_height,
    ]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_sam3(confidence_threshold: float):
    try:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise RuntimeError(
            "SAM 3 is not installed. Follow the GPU setup in PROTOTYPE.md, then rerun segment."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("The official SAM 3 image pipeline requires a CUDA-capable GPU.")
    model = build_sam3_image_model(device="cuda")
    return Sam3Processor(model, device="cuda", confidence_threshold=confidence_threshold)


def run_sam3(
    manifest_path: Path,
    *,
    output_dir: Path,
    confidence_threshold: float = 0.3,
    max_images: int | None = None,
    max_boxes_per_image: int | None = None,
    detections_path: Path | None = None,
    include_overlapping_detections: bool = False,
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_dir = manifest_path.parent
    results_dir = output_dir
    masks_dir = results_dir / "masks"
    overlays_dir = results_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    processor = _load_sam3(confidence_threshold)
    results: list[dict[str, Any]] = []
    detection_images: dict[str, dict[str, Any]] = {}
    if detections_path:
        detection_payload = json.loads(Path(detections_path).read_text(encoding="utf-8"))
        detection_images = {
            item["image_uuid"]: item for item in detection_payload.get("images", [])
        }

    records = manifest["images"][:max_images]
    for record in records:
        with Image.open(sample_dir / record["local_file"]) as loaded:
            image = loaded.convert("RGB")
        state = processor.set_image(image)
        visual_masks: list[tuple[str, np.ndarray]] = []
        if detections_path:
            predictions = detection_images.get(record["uuid"], {}).get("predictions", [])
            if not include_overlapping_detections:
                predictions = [
                    item for item in predictions if item["status"] == "missing_annotation_candidate"
                ]
            boxes = [
                {
                    **item,
                    "uuid": item.get("prediction_id"),
                    "concept": item.get("label"),
                }
                for item in predictions
            ]
        else:
            boxes = record["bounding_boxes"]
        boxes = boxes[:max_boxes_per_image]
        for box in boxes:
            processor.reset_all_prompts(state)
            prompt = normalize_xywh(box, image.width, image.height)
            state = processor.add_geometric_prompt(state=state, box=prompt, label=True)
            scores = _as_numpy(state["scores"]).reshape(-1)
            masks = _as_numpy(state["masks"])
            if not len(scores):
                results.append(
                    {
                        "image_uuid": record["uuid"],
                        "box_uuid": box.get("uuid"),
                        "concept": box.get("concept"),
                        "prompt_xywh": [box[k] for k in ("x", "y", "width", "height")],
                        "status": "no_mask_above_threshold",
                    }
                )
                continue
            best = int(np.argmax(scores))
            mask = np.squeeze(masks[best]).astype(bool)
            filename = f"{record['uuid']}__{box.get('uuid') or best}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(masks_dir / filename)
            visual_masks.append((str(box.get("concept") or "unknown"), mask))
            results.append(
                {
                    "image_uuid": record["uuid"],
                    "box_uuid": box.get("uuid"),
                    "concept": box.get("concept"),
                    "prompt_xywh": [box[k] for k in ("x", "y", "width", "height")],
                    "sam_prompt_normalized_cxcywh": prompt,
                    "score": float(scores[best]),
                    "mask_pixels": int(mask.sum()),
                    "mask_file": f"masks/{filename}",
                    "status": "ok",
                }
            )
        if visual_masks:
            overlay_masks(image, visual_masks).save(overlays_dir / f"{record['uuid']}.jpg", quality=92)

    result_path = results_dir / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "detections": str(detections_path) if detections_path else None,
                "confidence_threshold": confidence_threshold,
                "method": (
                    "SAM 3 positive visual box prompt on YOLO review candidates"
                    if detections_path and not include_overlapping_detections
                    else "SAM 3 positive visual box prompt on YOLO detections"
                    if detections_path
                    else "SAM 3 positive visual box prompt on each existing FathomNet box"
                ),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result_path
