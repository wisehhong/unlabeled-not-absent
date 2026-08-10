import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from unlabeled_not_absent.fathomnet import FathomNetError, download_image, image_suffix
from unlabeled_not_absent.sam3_runner import normalize_xywh
from unlabeled_not_absent.sampling import (
    benthic_evidence,
    build_manifest,
    collect_candidates,
    write_sample,
)
from unlabeled_not_absent.visualization import create_review_artifacts
from unlabeled_not_absent.yolo_runner import (
    box_iou_xywh,
    classify_predictions,
    prediction_records,
    run_yolo,
)


class _ArrayValue:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class PrototypeTests(unittest.TestCase):
    def test_normalize_xywh(self):
        prompt = normalize_xywh({"x": 10, "y": 20, "width": 30, "height": 40}, 100, 200)
        self.assertEqual(prompt, [0.25, 0.2, 0.3, 0.2])

    def test_benthic_evidence_is_explicit(self):
        image = {"boundingBoxes": [{"concept": "brittle star"}, {"concept": "bony fish"}]}
        self.assertEqual(benthic_evidence(image), ["brittle star"])

    def test_unknown_url_suffix_is_safe(self):
        self.assertEqual(image_suffix("https://example.test/image?id=1"), ".jpg")
        self.assertEqual(image_suffix("https://example.test/image.PNG?q=1"), ".png")

    @patch("unlabeled_not_absent.fathomnet._request", return_value=b"downloaded image")
    def test_download_image_verifies_sha256_before_replacing(self, request):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "sample.jpg"
            expected = hashlib.sha256(b"downloaded image").hexdigest()
            download_image(
                "https://example.test/sample.jpg",
                destination,
                expected_sha256=expected,
            )
            self.assertEqual(destination.read_bytes(), b"downloaded image")
            request.assert_called_once_with("https://example.test/sample.jpg", timeout=90)

    @patch("unlabeled_not_absent.fathomnet._request", return_value=b"wrong image")
    def test_download_image_rejects_sha256_mismatch(self, request):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "sample.jpg"
            with self.assertRaises(FathomNetError):
                download_image(
                    "https://example.test/sample.jpg",
                    destination,
                    expected_sha256=hashlib.sha256(b"expected image").hexdigest(),
                )
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".jpg.part").exists())

    def test_box_iou_xywh(self):
        first = {"x": 0, "y": 0, "width": 10, "height": 10}
        second = {"x": 5, "y": 0, "width": 10, "height": 10}
        self.assertAlmostEqual(box_iou_xywh(first, second), 1 / 3)

    def test_classify_predictions_flags_only_low_overlap(self):
        predictions = [
            {"x": 10, "y": 20, "width": 30, "height": 40, "label": "fish"},
            {"x": 70, "y": 70, "width": 10, "height": 10, "label": "fish"},
        ]
        annotations = [
            {"uuid": "box-1", "concept": "bony fish", "x": 10, "y": 20, "width": 30, "height": 40}
        ]
        classified = classify_predictions(predictions, annotations, overlap_threshold=0.1)
        self.assertEqual(classified[0]["status"], "overlaps_existing_annotation")
        self.assertEqual(classified[0]["matched_annotation_uuid"], "box-1")
        self.assertEqual(classified[1]["status"], "missing_annotation_candidate")

    def test_prediction_records_clips_to_image(self):
        boxes = type(
            "Boxes",
            (),
            {
                "xyxy": _ArrayValue([[-5, 10, 120, 90]]),
                "conf": _ArrayValue([0.87654321]),
                "cls": _ArrayValue([2]),
            },
        )()
        result = type("Result", (), {"boxes": boxes, "names": {2: "fish"}})()
        records = prediction_records(result, 100, 80)
        self.assertEqual(records[0]["label"], "fish")
        self.assertEqual(records[0]["confidence"], 0.876543)
        self.assertEqual(
            [records[0][key] for key in ("x", "y", "width", "height")],
            [0.0, 10.0, 100.0, 70.0],
        )

    @patch("unlabeled_not_absent.yolo_runner._load_yolo")
    def test_run_yolo_writes_results_and_review_artifacts(self, load_yolo):
        boxes = type(
            "Boxes",
            (),
            {
                "xyxy": _ArrayValue([[70, 70, 90, 90]]),
                "conf": _ArrayValue([0.9]),
                "cls": _ArrayValue([0]),
            },
        )()
        result = type("Result", (), {"boxes": boxes, "names": {0: "fish"}})()
        detector = load_yolo.return_value
        detector.predict.return_value = [result]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (100, 100), "navy").save(images / "sample.jpg")
            manifest = {
                "images": [
                    {
                        "uuid": "sample",
                        "local_file": "images/sample.jpg",
                        "bounding_boxes": [
                            {"uuid": "box-1", "concept": "sponge", "x": 10, "y": 10, "width": 20, "height": 20}
                        ],
                    }
                ]
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result_path = run_yolo(
                manifest_path, output_dir=root / "yolo", model="fake.pt"
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["missing_annotation_candidates"], 1)
            self.assertTrue((root / "yolo" / "overlays" / "sample.jpg").exists())
            self.assertTrue((root / "yolo" / "candidate_contact_sheet.jpg").exists())

    @patch("unlabeled_not_absent.sampling.fetch_image_page")
    def test_collect_candidates_filters_non_benthic(self, fetch):
        fetch.return_value = {
            "content": [
                self._image("one", "bony fish"),
                self._image("two", "sea fan"),
            ]
        }
        selected, pages = collect_candidates([7], page_size=2, limit=2, habitat="benthic")
        self.assertEqual([item["uuid"] for item in selected], ["two"])
        self.assertEqual(pages, [7])

    @patch("unlabeled_not_absent.sampling.download_image")
    def test_write_sample_reuses_image_when_sha256_matches(self, download):
        payload = b"already downloaded"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "sample.jpg"
            image_path.parent.mkdir()
            image_path.write_bytes(payload)
            manifest = {
                "images": [{
                    "uuid": "sample",
                    "source_url": "https://example.test/sample.jpg",
                    "local_file": "images/sample.jpg",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }]
            }
            with self.assertLogs("unlabeled_not_absent.sampling", level="INFO") as logs:
                write_sample(manifest, root)
            download.assert_not_called()
            self.assertTrue(any("SHA-256 matched" in message for message in logs.output))

    @patch("unlabeled_not_absent.sampling.download_image")
    def test_write_sample_redownloads_image_when_sha256_mismatches(self, download):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "sample.jpg"
            image_path.parent.mkdir()
            image_path.write_bytes(b"corrupt")
            manifest = {
                "images": [{
                    "uuid": "sample",
                    "source_url": "https://example.test/sample.jpg",
                    "local_file": "images/sample.jpg",
                    "sha256": hashlib.sha256(b"expected").hexdigest(),
                }]
            }
            write_sample(manifest, root)
            download.assert_called_once_with(
                "https://example.test/sample.jpg",
                image_path,
                expected_sha256=hashlib.sha256(b"expected").hexdigest(),
            )

    def test_manifest_and_contact_sheet(self):
        image = self._image("sample", "sponge")
        image["_benthicEvidence"] = ["sponge"]
        manifest = build_manifest([image], pages=[10], page_size=1, habitat="benthic")
        self.assertEqual(manifest["images"][0]["habitat_review"]["status"], "unreviewed_benthic_candidate")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / manifest["images"][0]["local_file"]
            image_path.parent.mkdir()
            Image.new("RGB", (100, 100), "navy").save(image_path)
            create_review_artifacts(manifest, root)
            self.assertTrue((root / "contact_sheet.jpg").exists())

    @staticmethod
    def _image(uuid, concept):
        return {
            "uuid": uuid,
            "url": f"https://example.test/{uuid}.jpg",
            "width": 100,
            "height": 100,
            "valid": True,
            "sha256": hashlib.sha256(uuid.encode()).hexdigest(),
            "boundingBoxes": [
                {
                    "uuid": f"box-{uuid}",
                    "concept": concept,
                    "x": 10,
                    "y": 20,
                    "width": 30,
                    "height": 40,
                    "reviewState": "VERIFIED",
                    "observer": "test",
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
