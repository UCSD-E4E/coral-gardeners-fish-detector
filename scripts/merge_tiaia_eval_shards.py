from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from coral_fish_pipeline.evaluation.metrics import precision_recall_f1


PER_IMAGE_FIELDS = ["image", "gt_count", "pred_count", "tp", "fp", "fn", "precision", "recall", "f1"]
MATCHED_CROP_FIELDS = [
    "image",
    "crop_id",
    "crop_path",
    "yolo_class",
    "matched_iou",
    "gt_box_xyxy",
    "pred_box_xyxy",
    "acceptable_species",
    "detection_score",
]
CLASSIFICATION_FIELDS = [
    "image",
    "crop_path",
    "yolo_class",
    "matched_iou",
    "predicted_species",
    "confidence",
    "top1_correct",
    "top5_correct",
    "acceptable_species",
    "top5",
    "model_id",
    "gt_box_xyxy",
    "pred_box_xyxy",
]


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def child_eval_dirs(root: Path) -> list[Path]:
    if (root / "metrics.json").exists():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "metrics.json").exists())


def merge(inputs: list[Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    eval_dirs: list[Path] = []
    for path in inputs:
        eval_dirs.extend(child_eval_dirs(path))
    if not eval_dirs:
        raise FileNotFoundError("No input directories with metrics.json found")

    per_image_rows: list[dict[str, Any]] = []
    matched_crop_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    total_images = 0
    totals = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}

    for eval_dir in eval_dirs:
        metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
        total_images += int(metrics.get("images_evaluated", 0))
        totals["tp"] += int(metrics.get("true_positives", 0))
        totals["fp"] += int(metrics.get("false_positives", 0))
        totals["fn"] += int(metrics.get("false_negatives", 0))
        totals["gt"] += int(metrics.get("total_ground_truth", 0))
        totals["pred"] += int(metrics.get("total_predictions", 0))
        per_image_rows.extend(read_csv_rows(eval_dir / "per_image_metrics.csv"))
        matched_crop_rows.extend(read_csv_rows(eval_dir / "classification" / "matched_crops.csv"))
        classification_rows.extend(read_csv_rows(eval_dir / "classification" / "matched_classifications.csv"))

    summary = precision_recall_f1(totals["tp"], totals["fp"], totals["fn"])
    merged_metrics: dict[str, Any] = {
        "inputs": [str(p) for p in eval_dirs],
        "images_evaluated": total_images,
        "total_ground_truth": totals["gt"],
        "total_predictions": totals["pred"],
        "true_positives": totals["tp"],
        "false_positives": totals["fp"],
        "false_negatives": totals["fn"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "f1": summary["f1"],
    }

    write_csv(output / "merged_per_image_metrics.csv", per_image_rows, PER_IMAGE_FIELDS)
    write_csv(output / "merged_matched_crops.csv", matched_crop_rows, MATCHED_CROP_FIELDS)
    if classification_rows:
        write_csv(output / "merged_matched_classifications.csv", classification_rows, CLASSIFICATION_FIELDS)
        top1 = sum(1 for row in classification_rows if bool_value(row.get("top1_correct")))
        top5 = sum(1 for row in classification_rows if bool_value(row.get("top5_correct")))
        count = len(classification_rows)
        cls_metrics = {
            "matched_detections_classified": count,
            "top1_correct": top1,
            "top1_accuracy_on_matched_detections": top1 / count if count else 0.0,
            "top5_correct": top5,
            "top5_accuracy_on_matched_detections": top5 / count if count else 0.0,
            "end_to_end_correct_top1": top1,
            "end_to_end_accuracy_top1_over_gt": top1 / totals["gt"] if totals["gt"] else 0.0,
            "end_to_end_correct_top5": top5,
            "end_to_end_accuracy_top5_over_gt": top5 / totals["gt"] if totals["gt"] else 0.0,
        }
        merged_metrics["classification"] = cls_metrics
        (output / "merged_classification_metrics.json").write_text(json.dumps(cls_metrics, indent=2), encoding="utf-8")

    (output / "merged_metrics.json").write_text(json.dumps(merged_metrics, indent=2), encoding="utf-8")
    lines = [
        "Merged Tiaia Evaluation",
        "=======================",
        f"Input dirs: {len(eval_dirs)}",
        f"Images evaluated: {total_images}",
        f"Ground-truth boxes: {totals['gt']}",
        f"Predicted boxes: {totals['pred']}",
        f"TP: {totals['tp']}",
        f"FP: {totals['fp']}",
        f"FN: {totals['fn']}",
        f"Precision: {summary['precision']:.4f}",
        f"Recall: {summary['recall']:.4f}",
        f"F1: {summary['f1']:.4f}",
    ]
    (output / "merged_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nMerged metrics written to: {output / 'merged_metrics.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Tiaia eval shard outputs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Eval output dirs or parent dirs containing eval outputs")
    parser.add_argument("--output", required=True, help="Merged output directory")
    args = parser.parse_args()
    merge([Path(p) for p in args.inputs], Path(args.output))


if __name__ == "__main__":
    main()
