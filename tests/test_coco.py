"""Tests for COCO normalization: id namespacing, class maps, provenance parsing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from una import coco  # noqa: E402


class TestParseProvenance:
    def test_soi_strips_part_suffix_so_one_dive_stays_together(self):
        base = "https://d2l7vcm2vanphr.cloudfront.net/full_images"
        part02 = f"{base}/SOI_FK190726_S0282_PART02/frame011076.png"
        part04 = f"{base}/SOI_FK190726_S0282_PART04/frame008228.png"

        source_a, dive_a = coco.parse_provenance(part02)
        source_b, dive_b = coco.parse_provenance(part04)

        assert source_a == source_b == "SOI"
        # Critical: both parts must share a dive id or a grouped split leaks
        # near-duplicate consecutive framegrabs across train and validation.
        assert dive_a == dive_b == "SOI_FK190726_S0282"

    def test_soi_without_part_suffix(self):
        url = ("https://d2l7vcm2vanphr.cloudfront.net/full_images/"
               "SOI_FK181210_S0205/FK181210_frame005065.png")
        assert coco.parse_provenance(url) == ("SOI", "SOI_FK181210_S0205")

    def test_noaa(self):
        url = ("https://d2l7vcm2vanphr.cloudfront.net/full_images/"
               "NOAA_OER_EX1702_DIVE10/EX1702_VID_frame007964.png")
        assert coco.parse_provenance(url) == ("NOAA", "NOAA_OER_EX1702_DIVE10")

    def test_hurl(self):
        url = ("https://hurlimage.soest.hawaii.edu/SupplementalPhotos/"
               "Localization_Scenes/D2-EX2301-07/Deep_Discoverer_23010007--45fd.jpg")
        assert coco.parse_provenance(url) == ("HURL", "HURL/D2-EX2301-07")

    def test_mbari_decodes_percent_encoded_platform(self):
        url = ("https://database.fathomnet.org/static/m3/framegrabs/"
               "Doc%20Ricketts/images/0536/03_38_06_10.png")
        source, dive = coco.parse_provenance(url)
        assert source == "MBARI"
        assert dive == "MBARI/Doc Ricketts/0536"

    def test_mbari_separates_dives_within_a_platform(self):
        base = "https://database.fathomnet.org/static/m3/framegrabs/Ventana/images"
        _, dive_a = coco.parse_provenance(f"{base}/3810/13_54_33_21.png")
        _, dive_b = coco.parse_provenance(f"{base}/4082/02_00_42_18.png")
        assert dive_a != dive_b

    def test_unknown_host_does_not_raise(self):
        assert coco.parse_provenance("https://example.com/a/b.png") == ("UNKNOWN", "UNKNOWN")


class TestCategoryMap:
    def test_sparse_ids_map_to_dense_contiguous_indices(self):
        # Mirrors the real file: ids span a range with gaps left by deleted classes.
        mapping = coco.CategoryMap(names={1: "urchin", 5: "crab", 41: "sea slug"})

        assert mapping.num_classes == 3
        assert mapping.to_index == {1: 0, 5: 1, 41: 2}
        assert mapping.to_category_id == [1, 5, 41]

    def test_round_trip_recovers_original_category_id(self):
        mapping = coco.CategoryMap(names={2: "a", 13: "b", 40: "c"})
        for category_id, index in mapping.to_index.items():
            assert mapping.to_category_id[index] == category_id

    def test_index_names_follow_index_order(self):
        mapping = coco.CategoryMap(names={41: "sea slug", 1: "urchin"})
        assert mapping.index_names() == ["urchin", "sea slug"]


@pytest.fixture
def tiny_coco(tmp_path: Path) -> Path:
    payload = {
        "images": [
            {
                "id": 7,
                "file_name": "aaaaaaaa-1111-2222-3333-444444444444.png",
                "coco_url": ("https://d2l7vcm2vanphr.cloudfront.net/full_images/"
                             "SOI_FK181210_S0205/frame1.png"),
                "flickr_url": ("https://d2l7vcm2vanphr.cloudfront.net/full_images/"
                               "SOI_FK181210_S0205/frame1.png"),
                "width": 3840,
                "height": 2160,
            },
            {
                "id": 9,
                "file_name": "bbbbbbbb-1111-2222-3333-444444444444.jpg",
                "coco_url": ("https://hurlimage.soest.hawaii.edu/SupplementalPhotos/"
                             "Localization_Scenes/D2-EX2301-07/img.jpg"),
                "flickr_url": "",
                "width": 1920,
                "height": 1080,
            },
        ],
        "annotations": [
            {"id": 100, "image_id": 7, "category_id": 5, "bbox": [10, 20, 30, 40], "iscrowd": 0},
            {"id": 101, "image_id": 7, "category_id": 41, "bbox": [1, 2, 3, 4], "iscrowd": 0},
        ],
        "categories": [{"id": 5, "name": "crab"}, {"id": 41, "name": "sea slug"}],
    }
    path = tmp_path / "train_dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestLoad:
    def test_image_ids_are_namespaced_by_split(self, tiny_coco: Path):
        dataset = coco.load("train", path=tiny_coco)
        assert {img.uid for img in dataset.images} == {"train:7", "train:9"}
        # The original id survives, because submissions need it verbatim.
        assert dataset.image("train:7").coco_image_id == 7

    def test_namespacing_prevents_cross_split_collision(self, tiny_coco: Path):
        train = coco.load("train", path=tiny_coco)
        test = coco.load("test", path=tiny_coco)
        assert not {i.uid for i in train.images} & {i.uid for i in test.images}

    def test_provenance_is_resolved_per_image(self, tiny_coco: Path):
        dataset = coco.load("train", path=tiny_coco)
        assert dataset.image("train:7").source == "SOI"
        assert dataset.image("train:9").source == "HURL"

    def test_annotations_are_grouped_by_image(self, tiny_coco: Path):
        dataset = coco.load("train", path=tiny_coco)
        assert len(dataset.annotations_for("train:7")) == 2
        assert dataset.annotations_for("train:9") == []

    def test_bbox_keeps_coco_xywh_convention(self, tiny_coco: Path):
        dataset = coco.load("train", path=tiny_coco)
        ann = dataset.annotations_for("train:7")[0]
        assert ann.bbox == (10.0, 20.0, 30.0, 40.0)
        assert ann.area == pytest.approx(1200.0)

    def test_dives_group_images(self, tiny_coco: Path):
        dataset = coco.load("train", path=tiny_coco)
        assert set(dataset.dives()) == {"SOI_FK181210_S0205", "HURL/D2-EX2301-07"}

    def test_orphan_annotation_is_rejected(self, tmp_path: Path):
        payload = {
            "images": [],
            "annotations": [
                {"id": 1, "image_id": 999, "category_id": 5, "bbox": [0, 0, 1, 1]}
            ],
            "categories": [],
        }
        path = tmp_path / "train_dataset.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown image"):
            coco.load("train", path=path)

    def test_zero_annotation_file_loads(self, tmp_path: Path):
        """test_dataset.json legitimately has no annotations; it must not error."""
        payload = {
            "images": [{
                "id": 1, "file_name": "x.png", "width": 10, "height": 10,
                "coco_url": "https://database.fathomnet.org/static/m3/framegrabs/"
                            "Ventana/images/3810/a.png",
            }],
            "annotations": [],
            "categories": [],
        }
        path = tmp_path / "test_dataset.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        dataset = coco.load("test", path=path)
        assert dataset.summary()["annotations"] == 0
