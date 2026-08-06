# Little Cayman species-detection evaluation

Benchmarks this repo's BioCLIP species classifier (and the full SAM3 → classify
pipeline) against **fishsense-lite human labels** on REEF imagery, for the
`little_cayman` site (the 13-species list in `resources/top25.yaml`).

Ground truth comes from the fishsense PostgreSQL database: fish **location** from
`headtaillabel` (snout/fork points), **species** from `specieslabel.content_of_image`,
and fish **selection** by the laser-on-fish criterion. Images are rectified from the
original `.ORF` raw files (the on-disk preprocess JPEGs are *not* lens-rectified).

## Results

Test set: **351 labeled fish**, 10 of the 13 site species (65 excluded: null
head/tail coords, or ambiguous multi-fish frames without a laser point).

| Condition | Top-1 | Top-5 |
|---|---|---|
| Classifier-only (GT crop), 13-species prior | **77.5%** | 92.6% |
| Classifier-only (GT crop), 69-species open-set | 72.4% | 85.8% |
| Full pipeline (SAM3→classify), 13-species prior | 56.7% end-to-end | 69.5% |
| Full pipeline, 69-species open-set | 52.7% end-to-end | 63.8% |

- SAM3 **detection recall = 78.1%** (274/351); on detected fish, classification is
  72.6% (13-prior) — within ~5 pts of GT crops, so detection is the dominant loss.
- **Black Grouper** fails end-to-end (0/25): under-detected and confused with other
  groupers. Confusions overall are taxonomically coherent (grouper↔grouper,
  parrotfish↔parrotfish).

See `report.html` (open in a browser) for tiles, per-species bars, and the
confusion matrix; `report_data.json` is its data.

## Rectification

`rectify_lib.py` reproduces `fishsense-core==2.4.1` exactly:
`rawpy.postprocess(gamma=(1,1), no_auto_bright, use_camera_wb, output_bps=16,
user_flip=0)` → auto-gamma from mean brightness → CLAHE (`equalize_adapthist`) →
BGR → `cv2.undistort(K, D)` (no new camera matrix, no crop). Label coordinates map
to the rectified full-res frame by **identity**. Full spec in `rectification_spec.md`.

## Reproduce

Everything runs in the project's Nix FHS shell. `run_tmux.sh` launches a command in
that sandbox inside a detached `tmux` session (survives SSH disconnect); it resolves
the FHS wrapper from the flake automatically. Extra deps beyond the main project:
`rawpy`, `scikit-image` (added to `pyproject.toml`), plus SAM3 (`install_sam3.sh`).

```bash
# 0. from repo root, one-time: nix develop; uv sync   (see top-level README)
cd eval/little_cayman

# 1. ground truth from a DB dump (uses nixpkgs#postgresql; no local pg client needed)
./extract_db.sh /path/to/fishsense/YYYY-MM-DDThh-mm-ssZ.dump   # -> gt/*.tsv, testset.ids

# 2. build the per-fish manifest (ORF_ROOT overrides the raw-file root)
#    default ORF_ROOT=~/mnt/fishsense_data/REEF/data
uv run python build_manifest.py                                # -> manifest.json

# 3. cache rectified frames from ORF (parallel; NWORKERS controls fan-out)
NWORKERS=48 uv run python cache_rectified.py                   # -> cache/frames/*.jpg
uv run python validate_overlay.py                             # optional: cache/qa overlays

# 4a. classifier-only eval (both label spaces)
uv run python eval_classifier.py                              # -> results_classifier_*.json

# 4b. full pipeline: SAM3 detect, then classify matched crops
uv run python full_pipeline_detect.py                         # -> detections.json  (1 GPU)
#   ...or shard across all GPUs:
NG=4 ./launch_detect_4gpu.sh && uv run python merge_detections.py
CUDA_VISIBLE_DEVICES=0 uv run python eval_full_pipeline.py    # -> results_fullpipe_*.json

# 5. aggregate + report data
uv run python aggregate_report.py                            # -> report_data.json
```

## Files

| File | Role |
|---|---|
| `extract_db.sh` | DB dump → `gt/*.tsv` + `testset.ids` (test-set selection) |
| `build_manifest.py` | join extracts → per-fish `manifest.json` |
| `rectify_lib.py` | ORF → rectified frame (fishsense-core repro) |
| `cache_rectified.py` | parallel rectify + cache all frames |
| `validate_overlay.py` | render label overlays for visual QA |
| `full_pipeline_detect.py` | SAM3 detect + match to labeled fish (`SHARD`/`NSHARDS` aware) |
| `eval_classifier.py` | classify GT crops, both label spaces |
| `eval_full_pipeline.py` | classify SAM3 matched crops; end-to-end metrics |
| `merge_detections.py` | merge per-GPU detection shards |
| `aggregate_report.py` | combine results → `report_data.json` |
| `run_tmux.sh` / `launch_detect_4gpu.sh` / `install_sam3.sh` | FHS/tmux + GPU launchers, SAM3 install |
| `species_sets.json` | the 13-species prior and 69-species open-set label spaces |
| `results_*.json`, `report_data.json`, `report.html` | committed results + report |

Regenerated data (`gt/`, `cache/`, `manifest.json`, `detections.json`, logs, images)
is git-ignored.
