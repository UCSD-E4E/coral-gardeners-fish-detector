from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from coral_fish_pipeline.utils.boxes import yolo_to_xyxy


@dataclass
class YoloBox:
    image_id: str
    class_id: int
    bbox_xyxy: list[float]


def get_split_dirs(dataset: str | Path, split: str) -> tuple[Path, Path]:
    root = Path(dataset)
    candidates = [
        (root / "images" / split, root / "labels" / split),
        (root / split / "images", root / split / "labels"),
    ]
    for img_dir, label_dir in candidates:
        if img_dir.exists() and label_dir.exists():
            return img_dir, label_dir
    raise FileNotFoundError(
        f"Could not find YOLO split directories for split={split}. Expected images/{split} and labels/{split}."
    )


def read_yolo_labels_for_image(image_path: str | Path, label_dir: str | Path) -> list[YoloBox]:
    image_path = Path(image_path)
    label_path = Path(label_dir) / f"{image_path.stem}.txt"
    if not label_path.exists():
        return []
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    boxes: list[YoloBox] = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Bad YOLO line in {label_path}:{line_no}: {line}")
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append(YoloBox(image_path.stem, cls, yolo_to_xyxy(cx, cy, w, h, img_w, img_h)))
    return boxes
