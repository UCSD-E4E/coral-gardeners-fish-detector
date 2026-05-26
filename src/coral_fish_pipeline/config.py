from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    default_path = project_root / "configs" / "default.yaml"
    if not default_path.exists():
        # Editable install from source tree: src/coral_fish_pipeline/config.py -> repo root is parents[2]
        default_path = Path.cwd() / "configs" / "default.yaml"
    cfg = load_yaml(default_path)
    if config_path:
        cfg = deep_update(cfg, load_yaml(config_path))
    return cfg


def ensure_output_dirs(output_dir: str | Path, create_artifact_dirs: bool = True) -> dict[str, Path]:
    out = Path(output_dir)
    dirs = {
        "root": out,
        "overlays": out / "overlays",
        "crops_raw": out / "crops" / "raw",
        "crops_masked": out / "crops" / "masked",
        "crops_by_species": out / "crops_by_species",
        "metadata": out / "metadata",
        "errors": out / "errors",
        "cache": out / ".cache",
    }
    required = ["root", "metadata"]
    if create_artifact_dirs:
        required = list(dirs)
    for key in required:
        path = dirs[key]
        path.mkdir(parents=True, exist_ok=True)
    return dirs
