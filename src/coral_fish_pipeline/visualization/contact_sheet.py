from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from coral_fish_pipeline.models import CropRecord, ClassificationResult


def make_contact_sheet(
    original_paths: list[str | Path],
    overlay_paths: list[str | Path],
    crops: list[CropRecord],
    results: list[ClassificationResult],
    output_path: str | Path,
    max_crops: int = 24,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    thumbs: list[Image.Image] = []
    for label, paths in [("Original", original_paths), ("Overlay", overlay_paths)]:
        for p in paths[:3]:
            img = Image.open(p).convert("RGB")
            img.thumbnail((360, 220))
            canvas = Image.new("RGB", (380, 260), "white")
            canvas.paste(img, ((380 - img.width) // 2, 25))
            d = ImageDraw.Draw(canvas)
            d.text((10, 5), f"{label}: {Path(p).name}", fill="black", font=font)
            thumbs.append(canvas)

    res_by_crop = {r.crop_id: r for r in results}
    for crop in crops[:max_crops]:
        p = Path(crop.raw_crop_path)
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        img.thumbnail((180, 160))
        canvas = Image.new("RGB", (220, 230), "white")
        canvas.paste(img, ((220 - img.width) // 2, 10))
        d = ImageDraw.Draw(canvas)
        r = res_by_crop.get(crop.crop_id)
        if r:
            text = f"{r.predicted_species}\n{r.confidence:.2f} {r.status}"
        else:
            text = crop.crop_id
        d.text((8, 175), text[:90], fill="black", font=font)
        thumbs.append(canvas)

    if not thumbs:
        Image.new("RGB", (600, 200), "white").save(output_path)
        return

    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + 40), "white")
    d = ImageDraw.Draw(sheet)
    d.text((10, 10), "Coral Fish Pipeline Contact Sheet", fill="black", font=title_font)
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * cell_w
        y = 40 + (i // cols) * cell_h
        sheet.paste(thumb, (x, y))
    sheet.save(output_path, quality=95)
