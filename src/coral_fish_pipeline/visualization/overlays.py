from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from coral_fish_pipeline.models import Detection


def draw_detections(image: Image.Image, detections: list[Detection], mode: str = "all") -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for det in detections:
        if mode == "accepted" and det.status != "accepted":
            continue
        if mode == "rejected" and det.status != "rejected":
            continue
        x1, y1, x2, y2 = det.bbox_xyxy
        color = (0, 255, 0, 220) if det.status == "accepted" else (255, 0, 0, 220)
        fill = (0, 255, 0, 35) if det.status == "accepted" else (255, 0, 0, 35)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.rectangle([x1, y1, x2, y2], fill=fill)
        label = det.det_id.split("_")[-1]
        if det.status == "rejected" and det.rejection_reason:
            label += f" {det.rejection_reason}"
        draw.text((x1 + 2, max(0, y1 - 18)), label, fill=color, font=font)
    return out


def save_overlays(image: Image.Image, image_id: str, detections: list[Detection], out_dir: str | Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for mode in ["all", "accepted", "rejected"]:
        p = out_dir / f"{image_id}_overlay_{mode}.jpg"
        draw_detections(image, detections, mode=mode).save(p, quality=95)
        paths[mode] = str(p)
    return paths
