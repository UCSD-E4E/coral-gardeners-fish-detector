from __future__ import annotations

from pathlib import Path


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name).strip("_") or "unknown"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for i in range(1, 100000):
        candidate = path.with_name(f"{stem}_{i:04d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique path for {path}")
