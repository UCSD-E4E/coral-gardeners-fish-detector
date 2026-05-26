from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

from coral_fish_pipeline.models import Detection
from coral_fish_pipeline.utils.masks import save_mask, mask_to_box


class MockSegmenter:
    """A dependency-free segmenter for testing the project wiring.

    This is not intended to be accurate. It finds dark-ish moving-object-like blobs and
    bright/red buoy-like blobs so the rest of the pipeline can be tested before SAM3 is installed.
    """

    def __init__(self, min_confidence: float = 0.25, max_detections_per_image: int = 100, **_: object) -> None:
        self.min_confidence = min_confidence
        self.max_detections = max_detections_per_image

    def predict(self, image: Image.Image, image_id: str, output_mask_dir: str | Path, prompts: list[str] | None = None) -> list[Detection]:
        import cv2

        output_mask_dir = Path(output_mask_dir)
        output_mask_dir.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(image.convert("RGB"))
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # Dark fish-like silhouettes plus red/orange buoys, for debugging accepted/rejected overlays.
        dark = (v < 95) & (s > 25)
        red_orange = (((h < 18) | (h > 170)) & (s > 70) & (v > 80))
        mask = (dark | red_orange).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for idx, c in enumerate(contours[: self.max_detections]):
            area = cv2.contourArea(c)
            if area < 40:
                continue
            m = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(m, [c], -1, 255, -1)
            bool_mask = m > 0
            box = mask_to_box(bool_mask)
            if box is None:
                continue
            det_id = f"{image_id}_mock_{idx:04d}"
            mask_path = output_mask_dir / f"{det_id}.png"
            save_mask(bool_mask, mask_path)
            detections.append(
                Detection(
                    image_id=image_id,
                    det_id=det_id,
                    bbox_xyxy=box,
                    mask_path=str(mask_path),
                    score=0.5,
                    prompt="mock",
                )
            )
        return detections
