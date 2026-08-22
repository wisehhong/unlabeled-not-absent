# FathomNet + YOLO + SAM 3 proof of concept

This prototype downloads a very small FathomNet sample, preserves its existing
bounding-box annotations, creates a contact sheet for habitat review, runs YOLO,
and compares its predictions with the existing boxes. High-confidence detections
with low class-agnostic overlap are written as possible missing-annotation
candidates. Candidate boxes can optionally be used as positive visual prompts for
SAM 3.

The prototype produces review candidates; it does **not** claim that unmatched
YOLO detections prove annotations are missing.

## 1. Fetch six likely benthic frames

Create a Python 3.12 environment and install this small package:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
una-poc fetch --pages 10,11,12 --page-size 10 --limit 6 --output data/poc
```

The command writes:

- `data/poc/manifest.json`: provenance, metadata, boxes, and explicit limitations
- `data/poc/images/`: downloaded originals
- `data/poc/previews/`: annotated previews
- `data/poc/contact_sheet.jpg`: the fastest way to confirm that frames are benthic

The API does not expose a dependable benthic/pelagic field. The default filter
therefore selects **benthic candidates** using labels such as `brittle star`,
`sea fan`, and `sponge`. Inspect the contact sheet and change each
`habitat_review.status` in the manifest to `confirmed_benthic` or `pelagic` before
treating habitat as research data.

For a reproducible wider sample, use deterministic random pages:

```powershell
una-poc fetch-random --seed 42 --page-count 12 --limit 6 --output data/random-poc
```

## 2. Run YOLO

Install the optional YOLO dependencies in the same environment:

```powershell
python -m pip install -e ".[yolo]"
```

Run a small smoke test with an Ultralytics checkpoint:

```powershell
una-poc detect `
  --manifest data/poc/manifest.json `
  --model yolo11n.pt `
  --max-images 2 `
  --output data/poc/yolo
```

Ultralytics downloads named checkpoints when they are not already available.
The generic `yolo11n.pt` model is trained on COCO and is only appropriate for
testing the pipeline. For research, pass a marine checkpoint fine-tuned or
validated for the selected FathomNet categories.

Useful controls include:

- `--confidence`: minimum detector confidence; default `0.25`
- `--nms-iou`: YOLO non-maximum-suppression IoU; default `0.7`
- `--overlap-iou`: minimum class-agnostic overlap with any existing annotation;
  detections below this value become review candidates; default `0.1`
- `--device`: an Ultralytics device such as `cpu`, `0`, or `cuda:0`
- `--max-images`: optional smoke-test limit

The output directory contains:

- `results.json`: checkpoint provenance, thresholds, predictions, maximum
  annotation IoU, matched annotation identifiers, candidate status, and summary
- `overlays/`: blue existing boxes, green overlapping detections, and red
  missing-annotation candidates
- `candidate_contact_sheet.jpg`: generated only when candidates exist

The original `manifest.json` is not changed.

YOLO support uses the third-party Ultralytics package. Review its current
licensing terms before redistributing the combined software or using it
commercially.

## 3. Optional: refine YOLO candidates with SAM 3 on a CUDA machine

Meta's official SAM 3 currently requires Python 3.12+, PyTorch 2.7+, a
CUDA-compatible GPU, access to the gated checkpoint on Hugging Face, and Hugging
Face authentication. Follow Meta's current installation instructions, install
this project into that same environment, then run:

```powershell
una-poc segment `
  --manifest data/poc/manifest.json `
  --detections data/poc/yolo/results.json `
  --output data/poc/sam3 `
  --max-images 2 `
  --max-boxes-per-image 4
```

Start with the limits shown: SAM 3 is an 848M-parameter model. The output includes
one binary mask per successful candidate prompt, an overlay per image, and
`results.json` with prompt coordinates, scores, mask areas, and provenance. When
`--detections` is omitted, the original behavior remains available and existing
FathomNet annotations are used as prompts. Add `--include-overlaps` to prompt all
YOLO detections rather than unmatched candidates only.

## 4. Verify the lightweight pipeline

The downloader and coordinate/visualization logic can be tested without SAM 3:

```powershell
python -m unittest discover -s tests -v
```

## Interpretation boundaries

- The benthic filter is a sampling aid, not habitat ground truth.
- FathomNet boxes may be incomplete. They are prompts here, not proof that every
  organism is annotated.
- A YOLO detection with low annotation overlap is a review candidate, not proof
  that an annotation is missing.
- YOLO/FathomNet overlap is class-agnostic because their label vocabularies may
  differ. The IoU threshold is an explicit audit parameter, not ground truth.
- Generic COCO weights do not validate marine-organism detection. Research claims
  require an appropriate checkpoint and a held-out, manually audited subset.
- A SAM mask is boundary refinement, not independent evidence of a missing label.
- Per the project feedback, SAM 3 used labeled FathomNet data during training.
  Report that possible non-independence when interpreting results.
- This prototype measures workflow feasibility and produces audit artifacts. It
  does not by itself support claims about detection accuracy, missing-label yield,
  or class balance.

## Primary references

- [FathomNet Python client](https://github.com/fathomnet/fathomnet-py)
- [FathomNet image API example](https://database.fathomnet.org/api/images?page=10&size=10)
- [Ultralytics documentation](https://docs.ultralytics.com/)
- [Meta's official SAM 3 repository](https://github.com/facebookresearch/sam3)
