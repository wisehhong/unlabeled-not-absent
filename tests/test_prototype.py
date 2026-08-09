import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from unlabeled_not_absent.fathomnet import image_suffix
from unlabeled_not_absent.sam3_runner import normalize_xywh
from unlabeled_not_absent.sampling import (
    benthic_evidence,
    build_manifest,
    collect_candidates,
)
from unlabeled_not_absent.visualization import create_review_artifacts


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

