#!/usr/bin/env python3
"""Build a top-N regional fish species prior from OBIS/GBIF observations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


OBIS_BASE = "https://api.obis.org/v3/occurrence"
GBIF_BASE = "https://api.gbif.org/v1/occurrence/search"
GBIF_SPECIES_MATCH_BASE = "https://api.gbif.org/v1/species/match"
USER_AGENT = "coral-fish-pipeline/top25-region-prior"

# Broad fish classes used for OBIS and GBIF coverage.
FISH_CLASSES = ["Actinopterygii", "Chondrichthyes", "Myxini"]


def http_get_json(base_url: str, params: dict[str, object], timeout: int = 30) -> dict:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{base_url}?{query}" if query else base_url
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def bbox_from_center(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    km_per_deg_lat = 111.32
    dlat = radius_km / km_per_deg_lat
    cos_lat = max(0.01, math.cos(math.radians(lat)))
    dlon = radius_km / (km_per_deg_lat * cos_lat)
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def bbox_to_wkt_polygon(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> str:
    return (
        f"POLYGON(({min_lon} {min_lat},"
        f"{min_lon} {max_lat},"
        f"{max_lon} {max_lat},"
        f"{max_lon} {min_lat},"
        f"{min_lon} {min_lat}))"
    )


def count_obis_species(geometry_wkt: str, max_records: int, batch_size: int, verbose: bool) -> Counter[str]:
    counts: Counter[str] = Counter()
    for fish_class in FISH_CLASSES:
        fetched = 0
        while fetched < max_records:
            size = min(batch_size, max_records - fetched)
            payload = http_get_json(
                OBIS_BASE,
                {
                    "geometry": geometry_wkt,
                    "scientificname": fish_class,
                    "size": size,
                    "offset": fetched,
                    "fields": "scientificName,scientificname",
                },
            )
            rows = payload.get("results", [])
            if not rows:
                break

            for row in rows:
                name = row.get("scientificName") or row.get("scientificname")
                if name:
                    counts[str(name)] += 1

            fetched += len(rows)
            if verbose:
                print(f"[OBIS {fish_class}] fetched {fetched} rows", file=sys.stderr)
            if len(rows) < size:
                break
    return counts


def gbif_usage_key_for_name(name: str) -> int | None:
    payload = http_get_json(GBIF_SPECIES_MATCH_BASE, {"name": name})
    key = payload.get("usageKey")
    return key if isinstance(key, int) else None


def count_gbif_species(geometry_wkt: str, max_records: int, batch_size: int, verbose: bool) -> Counter[str]:
    class_keys = [key for name in FISH_CLASSES if (key := gbif_usage_key_for_name(name))]
    if not class_keys:
        raise RuntimeError("Could not resolve GBIF class keys for fish taxa")

    counts: Counter[str] = Counter()
    for class_key in class_keys:
        fetched = 0
        while fetched < max_records:
            size = min(batch_size, max_records - fetched)
            payload = http_get_json(
                GBIF_BASE,
                {
                    "geometry": geometry_wkt,
                    "classKey": class_key,
                    "marine": "true",
                    "hasCoordinate": "true",
                    "limit": size,
                    "offset": fetched,
                },
            )
            rows = payload.get("results", [])
            for row in rows:
                name = row.get("species") or row.get("scientificName")
                if name:
                    counts[str(name)] += 1

            fetched += len(rows)
            if verbose:
                print(f"[GBIF classKey={class_key}] fetched {fetched} rows", file=sys.stderr)
            if payload.get("endOfRecords", False) or len(rows) < size:
                break
    return counts


def is_species_binomial(name: str) -> bool:
    parts = name.strip().split()
    if len(parts) != 2:
        return False
    genus, epithet = parts
    return genus[:1].isupper() and epithet.islower()


def make_region_key(raw: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def load_existing_regions(path: str | Path) -> dict[str, list[str]]:
    yaml_path = Path(path)
    if not yaml_path.exists():
        return {}

    regions: dict[str, list[str]] = {}
    current_region: str | None = None
    in_names = False
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.strip() == "regions:":
            continue
        if line.startswith("  ") and line.endswith(":") and not line.startswith("    "):
            current_region = line.strip()[:-1]
            regions.setdefault(current_region, [])
            in_names = False
            continue
        if current_region is None:
            continue
        if line.strip() == "names:":
            regions[current_region] = []
            in_names = True
            continue
        if line.strip().startswith("nc:"):
            in_names = False
            continue
        if in_names and line.strip().startswith("- "):
            regions[current_region].append(line.strip()[2:])
    return regions


def write_regions_yaml(path: str | Path, regions: dict[str, list[str]]) -> None:
    yaml_path = Path(path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["regions:"]
    for region_key, names in regions.items():
        lines.append(f"  {region_key}:")
        lines.append("    names:")
        for name in names:
            lines.append(f"      - {name}")
        lines.append(f"    nc: {len(names)}")
        lines.append("")
    yaml_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_csv(path: str | Path, ranked: Iterable[tuple[str, int]]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "species", "count"])
        for i, (species, count) in enumerate(ranked, start=1):
            writer.writerow([i, species, count])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or update regional top-N fish species candidates.")
    parser.add_argument("--lat", type=float, required=True, help="Center latitude")
    parser.add_argument("--lon", type=float, required=True, help="Center longitude")
    parser.add_argument("--radius-km", type=float, default=50.0, help="Search radius in km")
    parser.add_argument("--top-n", type=int, default=25, help="Number of species to write")
    parser.add_argument("--source", choices=["obis", "gbif", "both"], default="both", help="Observation source")
    parser.add_argument("--max-records", type=int, default=5000, help="Max records per source/class")
    parser.add_argument("--batch-size", type=int, default=300, help="Page size per API request")
    parser.add_argument("--out-yaml", default="resources/top25.yaml", help="Region YAML to update")
    parser.add_argument("--out-csv", help="Optional ranked CSV output")
    parser.add_argument("--region-key", help="Region key to update, such as moorea or fiji")
    parser.add_argument("--allow-higher-taxa", action="store_true", help="Include non-binomial taxa")
    parser.add_argument("--verbose", action="store_true", help="Print API progress to stderr")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    min_lat, max_lat, min_lon, max_lon = bbox_from_center(args.lat, args.lon, args.radius_km)
    geometry_wkt = bbox_to_wkt_polygon(min_lat, max_lat, min_lon, max_lon)

    merged_counts: Counter[str] = Counter()
    source_rows: dict[str, int] = {}

    try:
        if args.source in {"obis", "both"}:
            obis_counts = count_obis_species(geometry_wkt, args.max_records, args.batch_size, args.verbose)
            merged_counts.update(obis_counts)
            source_rows["obis"] = sum(obis_counts.values())
        if args.source in {"gbif", "both"}:
            gbif_counts = count_gbif_species(geometry_wkt, args.max_records, args.batch_size, args.verbose)
            merged_counts.update(gbif_counts)
            source_rows["gbif"] = sum(gbif_counts.values())
    except urllib.error.URLError as exc:
        print(f"Network/API error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3

    if not args.allow_higher_taxa:
        merged_counts = Counter({name: count for name, count in merged_counts.items() if is_species_binomial(name)})
    if not merged_counts:
        print("No species found for the specified region and filters.")
        return 0

    ranked = merged_counts.most_common(args.top_n)
    print(f"Top {len(ranked)} fish species near ({args.lat}, {args.lon}) within {args.radius_km} km")
    if source_rows:
        print("Sources: " + ", ".join(f"{source.upper()} rows={rows}" for source, rows in source_rows.items()))
    for i, (species, count) in enumerate(ranked, start=1):
        print(f"{i:2d}. {species} ({count})")

    if args.out_csv:
        write_csv(args.out_csv, ranked)
        print(f"\nWrote CSV: {args.out_csv}")

    region_key = make_region_key(args.region_key or f"region_{args.lat}_{args.lon}")
    regions = load_existing_regions(args.out_yaml)
    regions[region_key] = [species for species, _ in ranked]
    write_regions_yaml(args.out_yaml, regions)
    print(f"Wrote YAML: {args.out_yaml} (updated region: {region_key})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
