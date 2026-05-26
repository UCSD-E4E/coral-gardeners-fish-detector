#!/usr/bin/env bash
set -e

DATASET="data/Tiaia Fish Species v4.v2-rawimages.yolov8"
SPLIT="train"
START=0
END=100

mkdir -p outputs/ablation_tiaia_100/logs

run_eval () {
  NAME="$1"
  shift

  echo ""
  echo "===== RUNNING $NAME ====="

  rm -rf "outputs/ablation_tiaia_100/$NAME"

  PYTHONPATH=src python3 scripts/evaluate_tiaia_yolo.py \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --output "outputs/ablation_tiaia_100/$NAME" \
    --start-index "$START" \
    --end-index "$END" \
    --detect-only \
    --resume \
    "$@" \
    2>&1 | tee "outputs/ablation_tiaia_100/logs/${NAME}.log"
}

run_eval "baseline" \
  --preprocess none

run_eval "tiling_only" \
  --use-tiling \
  --tile-size 768 \
  --tile-overlap 0.25 \
  --preprocess none

run_eval "clahe_only" \
  --preprocess clahe_luminance

echo ""
echo "===== SUMMARY ====="

python3 - <<'PY'
from pathlib import Path
import json
import csv

root = Path("outputs/ablation_tiaia_100")
rows = []

for d in sorted(root.iterdir()):
    if not d.is_dir() or d.name == "logs":
        continue

    metrics_path = d / "metrics.json"
    if not metrics_path.exists():
        metrics_path = d / "metadata" / "metrics.json"

    if not metrics_path.exists():
        print(f"Missing metrics for {d.name}")
        continue

    m = json.loads(metrics_path.read_text())
    rows.append({
        "method": d.name,
        "images": m.get("images_evaluated"),
        "gt": m.get("total_ground_truth") or m.get("ground_truth_boxes"),
        "pred": m.get("total_predictions") or m.get("predicted_boxes"),
        "tp": m.get("true_positives") or m.get("TP") or m.get("tp"),
        "fp": m.get("false_positives") or m.get("FP") or m.get("fp"),
        "fn": m.get("false_negatives") or m.get("FN") or m.get("fn"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "f1": m.get("f1"),
    })

out = root / "summary.csv"
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "images", "gt", "pred", "tp", "fp", "fn", "precision", "recall", "f1"])
    writer.writeheader()
    writer.writerows(rows)

for r in rows:
    print(f"{r['method']:20s} precision={r['precision']:.4f} recall={r['recall']:.4f} f1={r['f1']:.4f} pred={r['pred']} fp={r['fp']} fn={r['fn']}")

print(f"\nSaved summary: {out}")
PY
