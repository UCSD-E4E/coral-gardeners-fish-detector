from __future__ import annotations

from coral_fish_pipeline.utils.boxes import iou_xyxy


def match_predictions_to_ground_truth(pred_boxes: list[list[float]], gt_boxes: list[list[float]], iou_threshold: float = 0.5):
    matches = []
    used_gt: set[int] = set()
    for pi, pred in enumerate(pred_boxes):
        best_iou = 0.0
        best_gi = None
        for gi, gt in enumerate(gt_boxes):
            if gi in used_gt:
                continue
            iou = iou_xyxy(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi is not None and best_iou >= iou_threshold:
            matches.append((pi, best_gi, best_iou))
            used_gt.add(best_gi)
    false_pos = [i for i in range(len(pred_boxes)) if i not in {m[0] for m in matches}]
    false_neg = [i for i in range(len(gt_boxes)) if i not in used_gt]
    return matches, false_pos, false_neg


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
