from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image

from coral_fish_pipeline.models import Detection
from coral_fish_pipeline.utils.boxes import box_area, box_aspect_ratio
from coral_fish_pipeline.utils.masks import load_mask, mask_circularity


def crop_array(arr: np.ndarray, box: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = arr.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return arr[y1:y2, x1:x2]


def red_orange_ratio(rgb_crop: np.ndarray) -> float:
    if rgb_crop.size == 0:
        return 0.0
    import cv2
    hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red_orange = (((h < 18) | (h > 170)) & (s > 70) & (v > 70))
    return float(red_orange.mean())


def has_vertical_dark_line_below(rgb: np.ndarray, box: list[float]) -> bool:
    import cv2
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    bw = max(1, x2 - x1)
    cx1 = max(0, int((x1 + x2) / 2 - bw * 0.2))
    cx2 = min(w, int((x1 + x2) / 2 + bw * 0.2))
    by1 = min(h, y2)
    by2 = min(h, int(y2 + max(15, (y2 - y1) * 2.5)))
    if by2 <= by1 or cx2 <= cx1:
        return False
    roi = rgb[by1:by2, cx1:cx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    dark = gray < 70
    # If a thin dark line appears below the detection, one column will have many dark pixels.
    col_density = dark.mean(axis=0) if dark.size else np.array([0])
    return bool(col_density.max() > 0.45 and dark.mean() < 0.35)


def reject_reason(det: Detection, image: Image.Image, cfg: dict) -> str | None:
    box = det.bbox_xyxy
    area = box_area(box)
    x1, y1, x2, y2 = box
    min_side = min(x2 - x1, y2 - y1)
    if area < float(cfg.get("min_box_area", 100)):
        return "too_small_area"
    if min_side < float(cfg.get("min_box_side", 8)):
        return "too_small_side"
    if box_aspect_ratio(box) > float(cfg.get("max_aspect_ratio", 8.0)):
        return "extreme_aspect_ratio"

    if cfg.get("enable_buoy_filter", True):
        rgb = np.asarray(image.convert("RGB"))
        rcrop = crop_array(rgb, box)
        rratio = red_orange_ratio(rcrop)
        circularity = 0.0
        if det.mask_path and Path(det.mask_path).exists():
            circularity = mask_circularity(load_mask(det.mask_path))
        rope = has_vertical_dark_line_below(rgb, box) if cfg.get("rope_check_enabled", True) else False
        if (
            rratio >= float(cfg.get("red_orange_ratio_threshold", 0.35))
            and circularity >= float(cfg.get("circularity_threshold", 0.75))
        ) or (rratio >= 0.25 and rope):
            return "likely_red_buoy"
    return None


def postprocess_detections(detections: list[Detection], image: Image.Image, cfg: dict) -> list[Detection]:
    processed: list[Detection] = []
    for det in detections:
        reason = reject_reason(det, image, cfg)
        if reason:
            processed.append(replace(det, status="rejected", rejection_reason=reason))
        else:
            processed.append(replace(det, status="accepted", rejection_reason=None))
    return processed
