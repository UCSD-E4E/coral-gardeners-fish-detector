#!/usr/bin/env bash
set -e

cd /mnt/c/Users/popov/Desktop/coral-fish-pipeline/coral-fish-pipeline

source /home/popovic/coral-fish-pipeline/.venv/bin/activate

export PYTHONPATH="$PWD/src"
export HF_HOME="/home/popovic/coral-fish-pipeline/models/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET="data/Tiaia Fish Species v4.v2-rawimages.yolov8"
SPLIT="train"
PROMPT="fish"
REGION="moorea"
CLASS_MAP="resources/tiaia_class_map.yaml"
BIOCLIP_MODEL="hf-hub:imageomics/bioclip-2.5-vith14"

mkdir -p outputs/eval_parts/logs

echo "===== PRECHECK ====="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
python3 -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 OK')"
python3 -c "import coral_fish_pipeline; print('pipeline OK')"

echo "===== STAGE 1: SAM3 DETECTION IN CHUNKS ====="

STARTS=(0 250 500 750 1000 1250 1500)
ENDS=(250 500 750 1000 1250 1500 1778)

for i in "${!STARTS[@]}"; do
  PART=$(printf "part_%03d" "$i")
  START=${STARTS[$i]}
  END=${ENDS[$i]}

  echo ""
  echo "===== DETECTION $PART: images $START to $END ====="

  PYTHONPATH=src python3 scripts/evaluate_tiaia_yolo.py \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --output "outputs/eval_parts/$PART" \
    --prompt "$PROMPT" \
    --start-index "$START" \
    --end-index "$END" \
    --detect-only \
    --resume \
    2>&1 | tee "outputs/eval_parts/logs/${PART}_detect.log"
done

echo ""
echo "===== STAGE 2: MERGE DETECTION SHARDS ====="

if [ -f scripts/merge_tiaia_eval_shards.py ]; then
  PYTHONPATH=src python3 scripts/merge_tiaia_eval_shards.py \
    --inputs \
      outputs/eval_parts/part_000 \
      outputs/eval_parts/part_001 \
      outputs/eval_parts/part_002 \
      outputs/eval_parts/part_003 \
      outputs/eval_parts/part_004 \
      outputs/eval_parts/part_005 \
      outputs/eval_parts/part_006 \
    --output outputs/eval_tiaia_train_sam3_merged \
    2>&1 | tee outputs/eval_parts/logs/merge_detection.log
else
  echo "No merge script found. Skipping detection merge."
fi

echo ""
echo "===== STAGE 3: BIOCLIP 2.5 CLASSIFICATION FROM SAVED CROPS ====="

for i in "${!STARTS[@]}"; do
  PART=$(printf "part_%03d" "$i")
  CROP_CSV="outputs/eval_parts/$PART/classification/matched_crops.csv"

  if [ ! -f "$CROP_CSV" ]; then
    echo "Skipping $PART classification because $CROP_CSV does not exist"
    continue
  fi

  echo ""
  echo "===== CLASSIFICATION $PART with $BIOCLIP_MODEL ====="

  PYTHONPATH=src python3 scripts/evaluate_tiaia_yolo.py \
    --dataset "$DATASET" \
    --output "outputs/eval_parts/${PART}_bioclip25" \
    --matched-crops-csv "$CROP_CSV" \
    --eval-classification \
    --region "$REGION" \
    --class-map "$CLASS_MAP" \
    --bioclip-model "$BIOCLIP_MODEL" \
    2>&1 | tee "outputs/eval_parts/logs/${PART}_classify_bioclip25.log"
done

echo ""
echo "===== STAGE 4: MERGE CLASSIFICATION SHARDS ====="

if [ -f scripts/merge_tiaia_eval_shards.py ]; then
  PYTHONPATH=src python3 scripts/merge_tiaia_eval_shards.py \
    --inputs \
      outputs/eval_parts/part_000_bioclip25 \
      outputs/eval_parts/part_001_bioclip25 \
      outputs/eval_parts/part_002_bioclip25 \
      outputs/eval_parts/part_003_bioclip25 \
      outputs/eval_parts/part_004_bioclip25 \
      outputs/eval_parts/part_005_bioclip25 \
      outputs/eval_parts/part_006_bioclip25 \
    --output outputs/eval_tiaia_train_bioclip25_merged \
    2>&1 | tee outputs/eval_parts/logs/merge_classification.log
else
  echo "No merge script found. Skipping classification merge."
fi

echo ""
echo "===== OVERNIGHT EVAL DONE ====="
echo "Detection merged output:"
echo "  outputs/eval_tiaia_train_sam3_merged"
echo "Classification merged output:"
echo "  outputs/eval_tiaia_train_bioclip25_merged"
echo ""
echo "Check logs:"
echo "  outputs/eval_parts/logs/"
