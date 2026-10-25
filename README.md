# Unlabeled Does Not Mean Absent: Improving Marine Organism Detection under Incomplete Deep Sea Image Annotations

This project studies object detection when the training annotations are known to be
incomplete: marine biologists label only the taxa relevant to their study, so an
unlabeled organism is usually *present but ignored*, not absent.

The work is grounded in **FathomNetCLEF2026** (CVPR FGVC13 / LifeCLEF 2026),
"Positive-Unlabeled Object Detection in Marine Images", which poses exactly this
problem on 6,463 training images with 22,225 boxes across 32 taxa.

## Repository layout

```text
fathomnet-2026/    Organizer-provided COCO files and official downloader (unmodified)
src/una/           Current research pipeline
prototype/         Original proof of concept, preserved and still runnable
tests/             Tests for the current pipeline
```

The earlier proof of concept lives in [prototype/PROTOTYPE.md](prototype/PROTOTYPE.md)
and remains installable as `una-poc`. It established the core idea this project builds
on: compare detections against existing boxes class-agnostically, and treat unmatched
detections as *review candidates rather than errors*.

## Data layout

Images are large (~55 GB) and are deliberately kept **outside** the repository so that
git worktrees stay disposable and the same checkout works on every machine. The data
root resolves in this order:

1. the `UNA_DATA_ROOT` environment variable
2. the `data_root` key in `una.config.json` at the repository root (gitignored)
3. `<repo>/data`

```jsonc
// una.config.json
{ "data_root": "C:/Users/you/Documents/Dev/fathomnet-data" }
```

## Getting the images

Downloading is a separate, resumable step, so that training and evaluation code never
mixes network failures into experimental results. Use the organizers' downloader:

```powershell
python -m pip install -r fathomnet-2026/requirements.txt
python fathomnet-2026/download.py fathomnet-2026/test_dataset.json  -o <data_root>/images/test
python fathomnet-2026/download.py fathomnet-2026/train_dataset.json -o <data_root>/images/train
```

On Windows, set `PYTHONUTF8=1` first; the downloader prints emoji that the default
cp1252 console encoding cannot represent.

## Commands

```powershell
python -m pip install -e .

una paths              # show every resolved path for this machine
una summary            # normalized statistics for both COCO splits
una doctor             # check downloaded images against the manifest
una doctor --verify    # additionally decode each image and check its dimensions
```

`una doctor` exits non-zero when a split is incomplete. This matters more than it
looks: a partial download is *biased*, not random, because large 4K SOI frames time out
while small HURL images succeed. Since institution correlates with habitat, camera and
annotation style, a silently partial download changes what the model learns while every
downstream number still looks plausible.

## YOLO baseline and Kaggle submission

A plain supervised baseline (unlabeled organisms treated as background) that exercises
the full submission flow. Install the extras first:
`python -m pip install -e .[yolo]` plus a CUDA build of PyTorch.

```powershell
una yolo-export        # resize train images to max side 1280, YOLO labels, dive-grouped val split
una train              # yolo11s, imgsz 1024, 30 epochs -> <data_root>/runs/yolo11s-baseline
una predict            # test split, conf=0.001, max_det=300 -> test_predictions.json
una submission         # -> submission.csv with original image_id and category_id

kaggle competitions submit -c fathomnet-2026 -f <data_root>/runs/yolo11s-baseline/submission.csv -m "yolo11s baseline"
```

The CSV columns are `annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_width,bbox_height,score`
with COCO pixel `xywh` boxes; the metric is mAP@[.5:.95]. The very low confidence
threshold is intentional, since low-score boxes can only add to mAP.

## Dataset handling notes

The organizer COCO files contain four traps that `src/una/coco.py` neutralizes:

| Trap | Consequence if ignored |
|---|---|
| `file_name` is a FathomNet UUID, never the URL basename | Images cannot be joined to their records |
| `image_id` collides across the two splits (1,405 ids) | Train and test annotations silently merge |
| `category_id` spans 1..41 with 9 gaps and is not a class index | Off-by-N class predictions in submissions |
| `flickr_url` duplicates `coco_url` byte for byte | Harmless, but ignored |

Provenance is recovered from image URLs because **a random train/validation split leaks
data**. One SOI dive alone is 18% of the training images as consecutive framegrabs, and
`_PART02` / `_PART04` directories are the same dive: the 21 source directories collapse
to 17 real dives. Splits must be grouped by dive.

Train and test are also a complete cross-institution domain shift — train is SOI, NOAA
and HURL; test is 100% MBARI — so validation on held-out training dives will read
optimistically compared with the competition test set.

## Licensing

YOLO support uses the third-party Ultralytics package. Review the
[Ultralytics license options](https://www.ultralytics.com/license) before
redistributing the combined software or using it commercially.

## Project Goals

This project is motivated by my long-standing interest in ocean exploration and marine life, as well as my growing interest in computer vision. Underwater images are collected by ROVs, AUVs, and cameras at a scale far beyond what experts can manually review. FathomNet helps address this by aggregating expert-labeled underwater images for AI research.

The goal of my project is to study marine organism detection in FathomNet images under incomplete annotation conditions. In underwater datasets, researchers often label only the organisms relevant to their scientific focus, leaving other visible organisms unlabeled. Images collected for one purpose may contain useful information for other studies, but that information can be difficult to reuse if it was not originally annotated.

This creates a problem for computer vision models. Standard object detection models often treat unlabeled regions as background, even when some may contain visible organisms. A correct detection of an unlabeled organism may therefore be counted as a false positive, reducing model performance and limiting dataset reuse for ocean science and biodiversity monitoring.

This project asks whether, in a limited subset of FathomNet images, an incomplete-annotation-aware workflow using YOLO detections, SAM-assisted refinement, and manual audit can flag plausible missing marine organism annotations that standard evaluation might treat as false positives.

To answer this question, I plan to develop a practical audit-and-detection workflow using limited organism categories and a manageable image subset. I will use a YOLO-based model as the baseline detector and compare its predictions with existing FathomNet annotations. High-confidence detections in unlabeled regions will be reviewed as possible missing annotation candidates rather than automatically treated as errors. I will use the Segment Anything Model as a support tool to refine candidate regions suggested by YOLO detections or existing boxes. The broader goal is not to correct all FathomNet annotations or build a new model from scratch. Instead, I hope to test whether existing computer vision tools can make marine image archives more reusable for future research.

In the future, I hope to study electrical engineering and computer science and work on ocean exploration, computer vision, robotics, and environmental sensing. RTP would help me move from independent projects into mentored original research using real marine image data.

## Methods

This project will use a manageable subset of publicly available FathomNet deep-sea images and bounding-box annotations. I will access image metadata and annotation information through the FathomNet Python client or API, then download the corresponding image files for analysis. To keep the project feasible, I will begin with benthic images, which are taken on or near the seafloor, rather than midwater or pelagic images, because midwater images often lack a visible bottom and may be more difficult for object boundary refinement and SAM-assisted segmentation.

I will begin with approximately 5-8 marine organism categories and 1000-2000 images, depending on class balance, data availability, and computing resources. I will organize the images into training, validation, and test sets. Before model training, I will also create a smaller manually audited subset of approximately 100-200 images to identify cases where visible organism-like regions appear unlabeled. Because marine organisms can be difficult to identify without expert knowledge, especially in deep-sea imagery, these cases will be treated as possible missing-annotation candidates for further review rather than definitive corrections.

I will use a YOLO-based object detection model as the baseline because it is widely used and accessible through Python or Google Colab. The model will be fine-tuned or evaluated using existing FathomNet bounding-box labels. This will show how standard object detection performs when available annotations are treated as ground truth.

Next, I will test an incomplete-annotation-aware workflow inspired by positive-unlabeled learning and sparsely annotated object detection. Instead of assuming all unlabeled regions are background, I will treat them as uncertain. I will compare YOLO detections with existing FathomNet annotations. High-confidence detections with low overlap against existing annotations will be reviewed as possible missing-annotation candidates rather than automatically counted as false positives.

I will use the Segment Anything Model as a support tool, not as the primary detector. SAM will refine candidate regions suggested by YOLO detections or existing boxes before manual review. However, SAM-generated masks will not be treated as independent proof of missing annotations. I will also note as a limitation that some SAM-family models may have had prior exposure to FathomNet-style or FathomNet-labeled data during training, so their performance on this dataset may not be fully independent.

Evaluation will include two parts. First, I will calculate standard object detection metrics, including precision, recall, mean Average Precision, and Intersection over Union, using the available FathomNet labels. Because the original annotations may be incomplete, standard mAP will be treated as a baseline metric rather than the only measure of success.

Second, on the manually audited subset, I will estimate candidate missing-annotation yield: the percentage of high-confidence detections in unlabeled regions that appear to contain visible organism-like regions after manual review. I will also record successful detections, false positives, difficult failure cases, and examples where expert review would be needed. The final result will be a practical analysis pipeline and research report, not a new model built from scratch.

## Resources

The main resources I need are realistic and accessible: public image data, open-source machine learning tools, cloud-based computing if needed, and research literature. I am not proposing to build a new dataset from scratch or collect underwater imagery myself. Instead, I plan to use existing marine image archives in a focused way and study whether a limited, reproducible workflow can help identify possible missing annotations. This project fits my future plans because I hope to study computer science and continue working where two of my passions, computer science and marine exploration, meet. RTP would provide the structure and mentorship needed to develop this interest into a more rigorous research project.

## Literature Cited

1. Katija, K., Orenstein, E., Schlining, B., Lundsten, L., Barnard, K., Sainz, G., Boulais, O., Cromwell, M., Butler, E., Woodward, B., and Bell, K. L. C., 2022, FathomNet: A global image database for enabling artificial intelligence in the ocean, Scientific Reports, v. 12, article 15914, doi:10.1038/s41598-022-19939-2.
2. Yang, Y., Liang, K. J., and Carin, L., 2020, Object detection as a positive-unlabeled problem, Proceedings of the 31st British Machine Vision Conference, doi:10.5244/C.34.74.
3. Suri, S., Rambhatla, S. S., Chellappa, R., and Shrivastava, A., 2023, SparseDet: Improving sparsely annotated object detection with pseudo-positive mining, Proceedings of the IEEE/CVF International Conference on Computer Vision, p. 6770-6781, doi:10.1109/ICCV51070.2023.00623.
4. Chen, G., Mao, Z., Shen, J., and Cheng, Z., 2025, Pseudo-label guided object detection in sparsely annotated underwater optical images, IEEE Transactions on Geoscience and Remote Sensing, v. 63, p. 1-21, doi:10.1109/TGRS.2025.3575841.
5. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollar, P., and Girshick, R., 2023, Segment anything, Proceedings of the IEEE/CVF International Conference on Computer Vision, p. 4015-4026.
6. Redmon, J., Divvala, S., Girshick, R., and Farhadi, A., 2016, You only look once: Unified, real-time object detection, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, p. 779-78.
