from __future__ import annotations

from pathlib import Path
import yaml


def load_regions(yaml_path: str | Path) -> dict[str, list[str]]:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Region YAML not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    regions_raw = data.get("regions", data)
    if not isinstance(regions_raw, dict):
        raise ValueError("Expected YAML with a 'regions' mapping")
    regions: dict[str, list[str]] = {}
    for region, info in regions_raw.items():
        if isinstance(info, dict):
            names = info.get("names")
        else:
            names = info
        if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
            raise ValueError(f"Region {region!r} does not contain a names list")
        regions[str(region)] = names
    return regions


def load_region_species(yaml_path: str | Path, region: str) -> list[str]:
    regions = load_regions(yaml_path)
    if region not in regions:
        available = ", ".join(sorted(regions))
        raise KeyError(f"Unknown region {region!r}. Available regions: {available}")
    return regions[region]
