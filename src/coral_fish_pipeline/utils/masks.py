from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


def load_mask(path: str | Path) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L"))
    return mask > 0


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    arr = (mask.astype(np.uint8) * 255)
    Image.fromarray(arr, mode="L").save(path)


def mask_to_box(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def mask_circularity(mask: np.ndarray) -> float:
    try:
        import cv2
        m = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            return 0.0
        return float(4.0 * np.pi * area / (perimeter * perimeter))
    except Exception:
        # Fallback: estimate circularity from bbox fill. Less accurate but safe.
        box = mask_to_box(mask)
        if box is None:
            return 0.0
        x1, y1, x2, y2 = box
        area = float(mask.sum())
        bbox_area = max(1.0, (x2 - x1) * (y2 - y1))
        return min(1.0, area / bbox_area)
