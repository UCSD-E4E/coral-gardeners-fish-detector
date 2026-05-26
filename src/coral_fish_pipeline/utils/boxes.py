from __future__ import annotations

import math
from typing import Any, Iterable


def clip_box_xyxy(box: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0.0, min(x1, width - 1))
    y1 = max(0.0, min(y1, height - 1))
    x2 = max(0.0, min(x2, width))
    y2 = max(0.0, min(y2, height))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return [x1, y1, x2, y2]


def expand_box_xyxy(box: Iterable[float], padding_percent: float, width: int, height: int, min_crop_size: int = 1) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * padding_percent
    pad_y = bh * padding_percent
    nx1, ny1, nx2, ny2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y

    # Enforce a minimum crop around the current box center.
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nw, nh = nx2 - nx1, ny2 - ny1
    if nw < min_crop_size:
        nx1, nx2 = cx - min_crop_size / 2.0, cx + min_crop_size / 2.0
    if nh < min_crop_size:
        ny1, ny2 = cy - min_crop_size / 2.0, cy + min_crop_size / 2.0
    return clip_box_xyxy([nx1, ny1, nx2, ny2], width, height)


def box_area(box: Iterable[float]) -> float:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_aspect_ratio(box: Iterable[float]) -> float:
    x1, y1, x2, y2 = map(float, box)
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return max(w / h, h / w)


def iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def non_max_suppression(detections: list[Any], iou_threshold: float) -> list[Any]:
    kept: list[Any] = []
    for det in sorted(detections, key=lambda d: float(getattr(d, "score", 1.0)), reverse=True):
        if any(iou_xyxy(getattr(det, "bbox_xyxy"), getattr(prev, "bbox_xyxy")) > iou_threshold for prev in kept):
            continue
        kept.append(det)
    return kept


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> list[float]:
    bw = w * img_w
    bh = h * img_h
    x1 = cx * img_w - bw / 2.0
    y1 = cy * img_h - bh / 2.0
    x2 = cx * img_w + bw / 2.0
    y2 = cy * img_h + bh / 2.0
    return clip_box_xyxy([x1, y1, x2, y2], img_w, img_h)
