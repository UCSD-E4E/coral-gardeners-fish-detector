# Coral Fish Pipeline

A reef fish computer vision pipeline using SAM3 for fish localization/cropping and BioCLIP 2.5 for region-aware species classification.

Pipeline:

```text
image/folder -> SAM3 detection -> postprocessing/NMS -> temporary crops -> BioCLIP 2.5 classification -> metadata/statistics
```

Default output is lightweight: metadata and stats only. Crops, masks, overlays, and contact sheets are optional debug artifacts.

## Project Overview

This project is a reproducible computer vision pipeline for reef fish imagery. It was built for the UCSD x National Geographic / Coral Gardeners class project and focuses on turning raw underwater images into structured fish detections, crops, species predictions, and evaluation metadata.

The pipeline separates the problem into two stages:

1. **Localization**: SAM3 is prompted to find fish in reef images, then detections are filtered with postprocessing and non-maximum suppression.
2. **Species classification**: accepted fish crops are classified with BioCLIP 2.5 or BioCLIP 2 using a region-specific top-25 species list.

The project is designed to be useful in two settings:

- **Field/project workflows**: process folders of reef images and produce compact CSV/JSON summaries.
- **Experiment workflows**: evaluate detection and classification behavior on labeled datasets such as Tiaia YOLO and classification-only GBIF/iNaturalist image folders.

By default, runs save lightweight metadata only. Crops, masks, overlays, and contact sheets can be enabled when visual inspection or demos are needed.

## Team

- Aleksa Popovic
- Ewan Shen
- Kaleigh Edusada
- Marlyn Arque Rupa
- Vibusha Vadivel

## Repository Guide

```text
class_materials/                 submitted class project documents and slides
configs/                         YAML defaults and experiment presets
data/                            local datasets and example images
external/                        local third-party checkouts, such as SAM3
models/                          local model caches; weights are not committed
outputs/                         generated runs, evals, metadata, and artifacts
resources/top25.yaml             region-specific BioCLIP species candidates
resources/tiaia_class_map.yaml   Tiaia broad-class to Latin-species mapping
scripts/                         evaluation, data download, and batch-run scripts
src/coral_fish_pipeline/         installable Python package
tests/                           unit and integration tests
```

Important package modules:

```text
segmentation/      SAM3 runner, mock segmenter, mask utilities, postprocessing
cropping/          crop creation from accepted detections
classification/    BioCLIP classifier, prompts, region species loader
evaluation/        detection and classification metrics
io/                image loading, YOLO label loading, metadata writers
visualization/     overlays and contact sheets
```

## Models, Data, and External Tools

Model weights and private credentials are intentionally not stored in this repository.

Required model/tool access:

- **SAM3**: installed separately from `facebookresearch/sam3`; real runs require access to the gated `facebook/sam3` Hugging Face model.
- **BioCLIP**: classification uses `hf-hub:imageomics/bioclip-2.5-vith14` by default, with `hf-hub:imageomics/bioclip-2` as a lower-memory fallback.
- **PyTorch**: install the build that matches your CUDA/CPU environment from the official PyTorch selector.

Supported data workflows:

- image or folder input for the main pipeline;
- Tiaia YOLO-format detection and classification evaluation;
- DeepFish SAM3 detection evaluation;
- GBIF/iNaturalist-style classification-only datasets for regional BioCLIP sanity checks.

Do not commit Hugging Face tokens, API keys, private dataset links, downloaded model weights, or unreleased/private data.

## Project Materials

This README is the main quick-start and reproduction document. Class deliverables are included in `class_materials/`:

- [Project Plan](class_materials/Coral%20Gardeners%20Project%20Plan.pdf)
- [Oral Update](class_materials/Coral%20Gardeners%20Oral%20Update.pptx)
- [Milestone Report](class_materials/Coral%20Gardeners%20Milestone%20Report.pdf)
- [Final Presentation](class_materials/Coral%20Gardeners%20Final%20Presentation.pptx)

The GitHub Wiki or a future `docs/` folder can hold longer writeups, additional figures, demo videos, and reproduction notes.

## Platform Notes

| Platform             | Recommendation                                                                          |
| -------------------- | --------------------------------------------------------------------------------------- |
| Windows native       | Not recommended for real SAM3 because Triton/SAM3 usually fails                         |
| Windows + WSL Ubuntu | Recommended for Windows users                                                           |
| Linux + NVIDIA GPU   | Best supported full-pipeline setup                                                      |
| macOS                | OK for BioCLIP/GBIF/dev tools, but real SAM3 may not work depending on official support |

For real SAM3 runs, use Linux or WSL Ubuntu with an NVIDIA GPU. Native Windows and macOS can still be useful for development, metadata tools, BioCLIP-only checks, and mock pipeline tests.

## Setup

From `<project-root>`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
pip install requests pyyaml tqdm pandas
```

Install PyTorch using the official selector for your machine:

https://pytorch.org/get-started/locally/

Install official SAM3 separately in the same environment:

```bash
mkdir -p external
git clone https://github.com/facebookresearch/sam3.git external/sam3
pip install -e external/sam3
```

Real SAM3 is not installed by this package. The `external/` folder is ignored by Git so the cloned SAM3 repo is not committed.

Set environment variables:

```bash
export PYTHONPATH="$PWD/src"
export HF_HOME="$PWD/models/hf_cache"
```

Log in to Hugging Face:

```bash
hf auth login
hf auth whoami
```

SAM3 uses the gated `facebook/sam3` Hugging Face repo. Accept/request access on Hugging Face before running real SAM3.

Test SAM3 model access:

```bash
python3 - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id="facebook/sam3", filename="config.json", token=True)
print("SAM3 config downloaded:", p)
PY
```

Basic checks:

```bash
python3 -c "import coral_fish_pipeline; print('pipeline OK')"
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
python3 -c "import open_clip; print('BioCLIP deps OK')"
```

Optional SAM3 check:

```bash
python3 -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 import OK')"
```

After editable install, you can use either command style:

```bash
PYTHONPATH=src python3 -m coral_fish_pipeline.cli --help
coral-fish-pipeline --help
```

## Run The Main Pipeline

The main pipeline accepts an image file or a folder. Folder inputs are scanned recursively for supported image extensions from `configs/default.yaml`.

Valid regions are keys under `regions:` in `resources/top25.yaml`. The bundled BioCLIP model IDs are:

```text
hf-hub:imageomics/bioclip-2.5-vith14
hf-hub:imageomics/bioclip-2
```

Original BioCLIP is intentionally blocked; use BioCLIP 2.5 or BioCLIP 2.

Lightweight default:

```bash
PYTHONPATH=src python3 -m coral_fish_pipeline.cli run \
  --input <path-to-images> \
  --region moorea \
  --output outputs/run_light \
  --segmenter sam3 \
  --bioclip-model hf-hub:imageomics/bioclip-2.5-vith14
```

Default output:

```text
outputs/run_light/metadata/
  detections.jsonl
  classifications.csv
  classifications.jsonl
  summary.json
  summary.txt
```

## Common Pipeline Flags

| Flag                             | Purpose                                                                       |
| -------------------------------- | ----------------------------------------------------------------------------- |
| `--input <path-to-images>`     | Image file or folder to process                                               |
| `--region <name>`              | Region key from `resources/top25.yaml`, used for BioCLIP species candidates |
| `--output <dir>`               | Output directory for metadata and optional artifacts                          |
| `--segmenter sam3`             | Use real SAM3 detection                                                       |
| `--segmenter mock`             | Use lightweight mock detection for wiring/tests                               |
| `--bioclip-model <id>`         | Choose BioCLIP model, usually `hf-hub:imageomics/bioclip-2.5-vith14`        |
| `--limit <n>`                  | Process only the first `n` images                                           |
| `--skip-classification`        | Run detection/cropping only; no BioCLIP classification                        |
| `--config <yaml>`              | Load config overrides, such as tiling settings                                |
| `--region-yaml <path>`         | Use a different region species YAML                                           |
| `--use-tiling`                 | Enable SAM3 tiled detection from the CLI                                      |
| `--tile-size <pixels>`         | Tile size when tiling is enabled                                              |
| `--tile-overlap <fraction>`    | Tile overlap when tiling is enabled                                           |
| `--preprocess clahe_luminance` | Apply CLAHE preprocessing to SAM3 detection input                             |

`--segmenter mock` is only for wiring and output tests. It is not a model-quality check.

## Optional Output Flags

| Flag                       | Saves                                |
| -------------------------- | ------------------------------------ |
| `--save-crops`           | crop images and `crops_by_species` |
| `--save-masks`           | SAM3 masks                           |
| `--save-overlays`        | visual overlays                      |
| `--save-contact-sheet`   | contact sheet                        |
| `--save-debug-artifacts` | all debug outputs                    |

Example:

```bash
PYTHONPATH=src python3 -m coral_fish_pipeline.cli run \
  --input <path-to-images> \
  --region moorea \
  --output outputs/run_debug \
  --segmenter sam3 \
  --bioclip-model hf-hub:imageomics/bioclip-2.5-vith14 \
  --save-debug-artifacts
```

## Modes

Default:

- no tiling
- no preprocessing
- fastest and cleanest

High-recall tiling:

```bash
PYTHONPATH=src python3 -m coral_fish_pipeline.cli run \
  --input <path-to-images> \
  --region moorea \
  --output outputs/run_high_recall \
  --segmenter sam3 \
  --config configs/high_recall_tiling.yaml \
  --bioclip-model hf-hub:imageomics/bioclip-2.5-vith14
```

Experimental CLAHE preprocessing:

```bash
PYTHONPATH=src python3 -m coral_fish_pipeline.cli run \
  --input <path-to-images> \
  --region moorea \
  --output outputs/run_clahe \
  --segmenter sam3 \
  --preprocess clahe_luminance
```

Preprocessing is applied only to the SAM3 detection input. BioCLIP still uses original crops.

## Evaluation

Evaluation defaults are lightweight too: metrics and CSV/JSON metadata are saved by default. Masks, overlays, and crops are opt-in. Use `--save-matched-crops` when you want to keep true-positive matched crops for classification reruns.

Tiaia YOLO detection eval:

```bash
PYTHONPATH=src python3 scripts/evaluate_tiaia_yolo.py \
  --dataset <path-to-tiaia-yolo-dataset> \
  --split train \
  --output outputs/eval_tiaia \
  --detect-only \
  --resume
```

Tiaia detection + classification eval:

```bash
PYTHONPATH=src python3 scripts/evaluate_tiaia_yolo.py \
  --dataset <path-to-tiaia-yolo-dataset> \
  --split train \
  --output outputs/eval_tiaia_cls \
  --eval-classification \
  --region moorea \
  --class-map resources/tiaia_class_map.yaml \
  --bioclip-model hf-hub:imageomics/bioclip-2.5-vith14 \
  --resume
```

DeepFish eval:

```bash
PYTHONPATH=src python3 scripts/evaluate_deepfish_sam3.py \
  --dataset <path-to-deepfish-dataset> \
  --split valid \
  --output outputs/eval_deepfish \
  --preprocess none
```

Useful Tiaia eval flags:

| Flag                                         | Purpose                                                         |
| -------------------------------------------- | --------------------------------------------------------------- |
| `--limit <n>`                              | Evaluate only the first `n` selected images                   |
| `--start-index <n>` / `--end-index <n>`  | Evaluate a slice of the split                                   |
| `--shard-index <i>` / `--num-shards <n>` | Split a run into shards                                         |
| `--resume`                                 | Reuse existing checkpoints when present                         |
| `--skip-existing`                          | Skip images with existing checkpoints                           |
| `--save-matched-crops`                     | Keep matched crop images for classification reruns              |
| `--classify-existing-crops <path>`         | Run classification from an existing matched-crops directory/CSV |
| `--matched-crops-csv <path>`               | Run classification from a specific `matched_crops.csv`        |

## GBIF/iNaturalist Classification Datasets

These are classification-only datasets. They do not test SAM3 because there are no boxes or masks.

Build or update a region top-25 fish prior from OBIS/GBIF observations:

```bash
PYTHONPATH=src python3 scripts/build_top25_region_prior.py \
  --lat -17.54 \
  --lon -149.83 \
  --radius-km 40 \
  --top-n 25 \
  --source both \
  --region-key moorea \
  --out-yaml resources/top25.yaml
```

The script reads any existing `resources/top25.yaml`, updates only the selected `--region-key`, and writes the same YAML format used by BioCLIP classification:

```yaml
regions:
  moorea:
    names:
      - Chaetodon auriga
    nc: 25
```

Use `--out-csv <path>` to also save the ranked source counts for inspection. By default, broader taxa are filtered out and only Latin binomial species names are written; pass `--allow-higher-taxa` to include genera or higher-level taxa.

Download Fiji:

```bash
PYTHONPATH=src python3 scripts/download_gbif_region_media.py \
  --region fiji \
  --species-yaml resources/top25.yaml \
  --out data/gbif_media \
  --max-per-species 25
```

Evaluate Fiji:

```bash
PYTHONPATH=src python3 scripts/evaluate_bioclip_folder_dataset.py \
  --dataset data/gbif_media/fiji \
  --region fiji \
  --output outputs/eval_bioclip_fiji_gbif \
  --bioclip-model hf-hub:imageomics/bioclip-2.5-vith14
```

Download/evaluate Gulf of Thailand by replacing:

```text
--region gulf_of_thailand
```

and using:

```text
data/gbif_media/gulf_of_thailand
```

GBIF/iNaturalist-style images may overlap with BioCLIP training data, so this is a region-specific sanity check, not a final unbiased benchmark.

## Tests

```bash
PYTHONPATH=src pytest tests -q --basetemp=.pytest-tmp
```

Tiny mock check:

```bash
PYTHONPATH=src python3 -m coral_fish_pipeline.cli run \
  --input data/examples \
  --region moorea \
  --output outputs/dev_mock_check \
  --segmenter mock \
  --skip-classification
```

## Common Issues

- `ModuleNotFoundError: triton`: use WSL/Linux for real SAM3.
- Hugging Face 401 for `facebook/sam3`: accept/request model access and run `hf auth login`.
- `ModuleNotFoundError: requests`: run `pip install requests pyyaml tqdm pandas`.
- CUDA out of memory: close GPU-heavy apps or use `hf-hub:imageomics/bioclip-2`.
- Need visual outputs: rerun with `--save-debug-artifacts`.
