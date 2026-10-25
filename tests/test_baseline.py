"""Tests for the YOLO baseline's pure helpers: split, labels, submission rows."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from una import baseline, coco  # noqa: E402


def _image(uid: str, dive: str, width: int = 100, height: int = 50) -> coco.ImageRecord:
    return coco.ImageRecord(
        uid=uid, coco_image_id=int(uid.split(":")[1]), split="train", file_name=f"{uid}.png",
        url="", width=width, height=height, source="SOI", dive=dive,
    )


def _dataset(dive_sizes: dict[str, int]) -> coco.Dataset:
    images, n = [], 0
    for dive, size in dive_sizes.items():
        for _ in range(size):
            n += 1
            images.append(_image(f"train:{n}", dive))
    return coco.Dataset("train", images, [], coco.CategoryMap(names={1: "a"}))


class TestSplitByDive:
    def test_dives_are_never_split(self):
        dataset = _dataset({"a": 5, "b": 3, "c": 7, "d": 2, "e": 4})
        val = baseline.split_by_dive(dataset, 0.3, seed=1)
        for dive, members in dataset.dives().items():
            inside = {img.uid in val for img in members}
            assert len(inside) == 1, dive

    def test_oversized_dive_is_not_forced_into_val(self):
        dataset = _dataset({"huge": 80, "s1": 5, "s2": 5, "s3": 5, "s4": 5})
        val = baseline.split_by_dive(dataset, 0.15, seed=0)
        assert 0 < len(val) <= 15

    def test_seed_is_deterministic(self):
        dataset = _dataset({str(i): i + 1 for i in range(10)})
        assert baseline.split_by_dive(dataset, 0.2, 3) == baseline.split_by_dive(dataset, 0.2, 3)


class TestYoloLabels:
    def test_normalizes_and_uses_dense_index(self):
        img = _image("train:1", "d", width=200, height=100)
        cats = coco.CategoryMap(names={5: "crab", 41: "slug"})
        ann = coco.AnnotationRecord(1, img.uid, 41, (20.0, 10.0, 40.0, 20.0))
        assert baseline.yolo_label_lines(img, [ann], cats) == [
            "1 0.200000 0.200000 0.200000 0.200000"
        ]

    def test_clips_boxes_and_drops_degenerate(self):
        img = _image("train:1", "d", width=100, height=100)
        cats = coco.CategoryMap(names={5: "crab"})
        overhang = coco.AnnotationRecord(1, img.uid, 5, (-10.0, 90.0, 30.0, 30.0))
        outside = coco.AnnotationRecord(2, img.uid, 5, (150.0, 0.0, 10.0, 10.0))
        assert baseline.yolo_label_lines(img, [overhang, outside], cats) == [
            "0 0.100000 0.950000 0.200000 0.100000"
        ]


class TestSubmissionRows:
    def test_maps_index_back_to_category_and_converts_to_xywh(self):
        cats = coco.CategoryMap(names={3: "a", 28: "b"})
        predictions = {"images": [{
            "image_id": 77, "width": 1920, "height": 1080,
            "cls": [1, 0], "xyxy": [[10, 20, 110, 220], [1900, 1000, 2000, 1200]],
            "score": [0.9, 0.01],
        }]}
        rows = baseline.submission_rows(predictions, cats)
        assert rows[0] == [1, 77, 28, 10, 20, 100, 200, 0.9]
        # Second box is clipped to the image bounds.
        assert rows[1] == [2, 77, 3, 1900, 1000, 20, 80, 0.01]

    def test_annotation_ids_are_unique(self):
        cats = coco.CategoryMap(names={1: "a"})
        predictions = {"images": [
            {"image_id": i, "width": 10, "height": 10, "cls": [0, 0],
             "xyxy": [[0, 0, 5, 5], [1, 1, 6, 6]], "score": [0.5, 0.4]}
            for i in range(3)
        ]}
        rows = baseline.submission_rows(predictions, cats)
        assert len({r[0] for r in rows}) == len(rows) == 6

    def test_zero_area_boxes_are_dropped(self):
        cats = coco.CategoryMap(names={1: "a"})
        predictions = {"images": [{
            "image_id": 1, "width": 10, "height": 10, "cls": [0],
            "xyxy": [[20, 20, 30, 30]], "score": [0.5],
        }]}
        assert baseline.submission_rows(predictions, cats) == []


def test_submission_columns_match_kaggle_spec():
    assert baseline.SUBMISSION_COLUMNS == (
        "annotation_id", "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_width", "bbox_height", "score",
    )


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_scores_stay_in_unit_interval(value):
    cats = coco.CategoryMap(names={1: "a"})
    predictions = {"images": [{"image_id": 1, "width": 10, "height": 10, "cls": [0],
                               "xyxy": [[0, 0, 5, 5]], "score": [value]}]}
    assert 0.0 <= baseline.submission_rows(predictions, cats)[0][7] <= 1.0
