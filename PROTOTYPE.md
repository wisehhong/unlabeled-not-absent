# FathomNet + SAM 3 proof of concept

This prototype downloads a very small FathomNet sample, preserves its existing
bounding-box annotations, creates a contact sheet for habitat review, and can
optionally use each existing box as a positive visual prompt for SAM 3.

It deliberately does **not** run YOLO or claim to find missing annotations yet.
The first question is narrower: *can SAM 3 turn FathomNet boxes into useful,
reviewable organism masks on a few likely benthic frames?*

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

## 2. Optional: run SAM 3 on a CUDA machine

Meta's official SAM 3 currently requires Python 3.12+, PyTorch 2.7+, a
CUDA-compatible GPU, access to the gated checkpoint on Hugging Face, and Hugging
Face authentication. Follow Meta's current installation instructions, install
this project into that same environment, then run:

```powershell
una-poc segment `
  --manifest data/poc/manifest.json `
  --output data/poc/sam3 `
  --max-images 2 `
  --max-boxes-per-image 4
```

Start with the limits shown: SAM 3 is an 848M-parameter model. The output includes
one binary mask per successful box prompt, an overlay per image, and `results.json`
with prompt coordinates, scores, mask areas, and provenance.

## 3. Verify the lightweight pipeline

The downloader and coordinate/visualization logic can be tested without SAM 3:

```powershell
python -m unittest discover -s tests -v
```

## Interpretation boundaries

- The benthic filter is a sampling aid, not habitat ground truth.
- FathomNet boxes may be incomplete. They are prompts here, not proof that every
  organism is annotated.
- A SAM mask is boundary refinement, not independent evidence of a missing label.
- Per the project feedback, SAM 3 used labeled FathomNet data during training.
  Report that possible non-independence when interpreting results.
- This prototype measures feasibility and produces examples. It does not yet
  support claims about detection accuracy, missing-label yield, or class balance.

## Primary references

- [FathomNet Python client](https://github.com/fathomnet/fathomnet-py)
- [FathomNet image API example](https://database.fathomnet.org/api/images?page=10&size=10)
- [Meta's official SAM 3 repository](https://github.com/facebookresearch/sam3)
