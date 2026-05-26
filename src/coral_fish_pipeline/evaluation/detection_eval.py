from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil

from PIL import Image

from coral_fish_pipeline.io.yolo_loader import get_split_dirs, read_yolo_labels_for_image
from coral_fish_pipeline.evaluation.metrics import match_predictions_to_ground_truth, precision_recall_f1
from coral_fish_pipeline.models import Detection
from coral_fish_pipeline.utils.boxes import iou_xyxy


def evaluate_detection_for_image(preds: list[Detection], gt_boxes: list[list[float]], iou_threshold: float) -> dict[str, Any]:
    accepted = [p for p in preds if p.status == "accepted"]
    pred_boxes = [p.bbox_xyxy for p in accepted]
    matches, fp_idx, fn_idx = match_predictions_to_ground_truth(pred_boxes, gt_boxes, iou_threshold)
    metrics = precision_recall_f1(len(matches), len(fp_idx), len(fn_idx))
    metrics.update({
        "tp": len(matches),
        "fp": len(fp_idx),
        "fn": len(fn_idx),
        "matches": matches,
        "false_positive_indices": fp_idx,
        "false_negative_indices": fn_idx,
    })
    return metrics


def summarize_eval(image_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(m["tp"]) for m in image_metrics)
    fp = sum(int(m["fp"]) for m in image_metrics)
    fn = sum(int(m["fn"]) for m in image_metrics)
    out = precision_recall_f1(tp, fp, fn)
    out.update({"tp": tp, "fp": fp, "fn": fn, "num_images": len(image_metrics), "ap50": out["precision"]})
    return out
