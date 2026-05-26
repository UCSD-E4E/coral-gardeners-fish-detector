from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

from coral_fish_pipeline.models import Detection, CropRecord
from coral_fish_pipeline.utils.boxes import expand_box_xyxy
from coral_fish_pipeline.utils.masks import load_mask


def _int_box(box: list[float]) -> tuple[int, int, int, int]:
    return tuple(int(round(v)) for v in box)  # type: ignore[return-value]


def create_crops(
    image: Image.Image,
    image_id: str,
    detections: list[Detection],
    raw_dir: str | Path,
    masked_dir: str | Path,
    padding_percent: float = 0.30,
    min_crop_size: int = 64,
    save_masked_crop: bool = True,
) -> list[CropRecord]:
    raw_dir = Path(raw_dir)
    masked_dir = Path(masked_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)
    w, h = image.size
    records: list[CropRecord] = []
    for det in detections:
        if det.status != "accepted":
            continue
        padded = expand_box_xyxy(det.bbox_xyxy, padding_percent, w, h, min_crop_size)
        x1, y1, x2, y2 = _int_box(padded)
        crop_id = det.det_id
        raw_crop = image.crop((x1, y1, x2, y2))
        raw_path = raw_dir / f"{crop_id}.jpg"
        raw_crop.save(raw_path, quality=95)

        masked_path: str | None = None
        if save_masked_crop and det.mask_path:
            mask = load_mask(det.mask_path)
            arr = np.asarray(image.convert("RGB")).copy()
            full_mask = mask.astype(bool)
            if full_mask.shape[:2] == arr.shape[:2]:
                arr[~full_mask] = 0
                masked_img = Image.fromarray(arr, mode="RGB").crop((x1, y1, x2, y2))
                mp = masked_dir / f"{crop_id}.png"
                masked_img.save(mp)
                masked_path = str(mp)

        records.append(
            CropRecord(
                crop_id=crop_id,
                image_id=image_id,
                det_id=det.det_id,
                raw_crop_path=str(raw_path),
                masked_crop_path=masked_path,
                bbox_original=[float(v) for v in det.bbox_xyxy],
                bbox_padded=[float(v) for v in padded],
            )
        )
    return records
