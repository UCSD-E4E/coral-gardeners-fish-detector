from __future__ import annotations

from pathlib import Path
from typing import Iterable
from PIL import Image


def list_images(input_path: str | Path, extensions: Iterable[str] | None = None, limit: int | None = None) -> list[Path]:
    path = Path(input_path)
    exts = {e.lower() for e in (extensions or [".jpg", ".jpeg", ".png", ".bmp", ".webp"])}
    if path.is_file():
        images = [path] if path.suffix.lower() in exts else []
    elif path.is_dir():
        images = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if limit is not None:
        images = images[:limit]
    return images


def load_image_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_id_from_path(path: str | Path) -> str:
    return Path(path).stem
