from __future__ import annotations

import numpy as np
from PIL import Image


def gray_world_white_balance(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    means = arr.reshape(-1, 3).mean(axis=0)
    grand = means.mean()
    scale = grand / np.maximum(means, 1e-6)
    out = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def clahe_luminance(image: Image.Image) -> Image.Image:
    import cv2
    arr = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb, mode="RGB")


def gamma_correction(image: Image.Image, gamma: float = 1.1) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    out = np.clip(arr ** (1.0 / gamma), 0, 1) * 255.0
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def apply_preprocessing(
    image: Image.Image,
    enabled: bool = False,
    method: str = "none",
    apply_to_segmentation: bool = True,
    apply_to_classification: bool = False,
    gamma: float = 1.1,
) -> Image.Image:
    if not enabled or method == "none":
        return image
    if method in {"gray_world", "gray_world_white_balance"}:
        return gray_world_white_balance(image)
    if method == "clahe_luminance":
        return clahe_luminance(image)
    if method in {"gamma", "gamma_correction"}:
        return gamma_correction(image, gamma=gamma)
    raise ValueError(f"Unknown preprocessing method: {method}")
