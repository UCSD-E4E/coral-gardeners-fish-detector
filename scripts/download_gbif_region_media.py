from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


GBIF_OCCURRENCE_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"
GULF_OF_THAILAND_WKT = "POLYGON((99.0 5.5,105.2 5.5,105.2 13.8,99.0 13.8,99.0 5.5))"
DEFAULT_LICENSES = ["CC0_1_0", "CC_BY_4_0", "CC_BY_NC_4_0"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MANIFEST_FIELDS = [
    "region",
    "species",
    "species_slug",
    "gbif_id",
    "occurrence_id",
    "image_url",
    "local_path",
    "license",
    "rights_holder",
    "publisher",
    "dataset_key",
    "references",
    "decimal_latitude",
    "decimal_longitude",
    "event_date",
    "downloaded",
    "skip_reason",
]
COUNTS_FIELDS = ["species", "requested_max", "downloaded_count", "skipped_count"]


def slug_species(species: str) -> str:
    return "_".join(species.strip().split())


def unslug_species(slug: str) -> str:
    return " ".join(slug.strip().split("_"))


def normalize_license(value: str | None) -> str:
    text = (value or "").strip().upper()
    if not text:
        return ""
    text = text.replace("HTTPS://CREATIVECOMMONS.ORG/LICENSES/", "")
    text = text.replace("HTTP://CREATIVECOMMONS.ORG/LICENSES/", "")
    text = text.replace("HTTPS://CREATIVECOMMONS.ORG/PUBLICDOMAIN/ZERO/", "CC0/")
    text = text.replace("HTTP://CREATIVECOMMONS.ORG/PUBLICDOMAIN/ZERO/", "CC0/")
    text = text.strip("/")
    text = text.replace("-", "_").replace("/", "_").replace(".", "_")
    if text.startswith(("BY_", "BY_NC_", "BY_SA_", "BY_ND_")):
        text = f"CC_{text}"
    aliases = {
        "CC0": "CC0_1_0",
        "CC0_1_0_LEGALCODE": "CC0_1_0",
        "CC_BY_4_0_LEGALCODE": "CC_BY_4_0",
        "CC_BY_NC_4_0_LEGALCODE": "CC_BY_NC_4_0",
    }
    return aliases.get(text, text)


def is_allowed_license(value: str | None, allowed: set[str]) -> bool:
    return normalize_license(value) in {normalize_license(item) for item in allowed}


def infer_extension(content_type: str | None, url: str) -> str | None:
    path = urlparse(url).path.lower()
    suffix = Path(path).suffix
    if suffix in IMAGE_EXTENSIONS:
        return ".jpg" if suffix == ".jpeg" else suffix
    content = (content_type or "").split(";")[0].strip().lower()
    if content == "image/jpeg":
        return ".jpg"
    if content == "image/png":
        return ".png"
    if content == "image/webp":
        return ".webp"
    return None


def load_region_species(path: Path, region: str) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    regions = data.get("regions", data)
    info = regions.get(region)
    if info is None:
        raise KeyError(f"Unknown region {region!r}. Available regions: {', '.join(sorted(regions))}")
    names = info.get("names") if isinstance(info, dict) else info
    if not isinstance(names, list):
        raise ValueError(f"Region {region!r} does not contain a species list")
    return [str(name) for name in names]


def region_params(region: str) -> dict[str, str]:
    if region == "fiji":
        return {"country": "FJ"}
    if region == "gulf_of_thailand":
        return {"geometry": GULF_OF_THAILAND_WKT}
    raise ValueError(f"Unsupported region: {region}")


def media_url(media: dict[str, Any]) -> str:
    return str(media.get("identifier") or media.get("references") or "").strip()


def query_gbif_records(
    species: str,
    region: str,
    source_dataset: str | None = None,
    limit: int = 300,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "scientificName": species,
        "mediaType": "StillImage",
        "hasCoordinate": "true",
        "occurrenceStatus": "PRESENT",
        "limit": min(limit, 300),
    }
    params.update(region_params(region))
    if source_dataset:
        params["datasetKey"] = source_dataset

    response = requests.get(GBIF_OCCURRENCE_SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return list(data.get("results", []))


def build_manifest_row(
    region: str,
    species: str,
    record: dict[str, Any],
    media: dict[str, Any],
    image_url: str,
    local_path: Path | None,
    downloaded: bool,
    skip_reason: str,
) -> dict[str, Any]:
    return {
        "region": region,
        "species": species,
        "species_slug": slug_species(species),
        "gbif_id": record.get("gbifID", ""),
        "occurrence_id": record.get("occurrenceID", ""),
        "image_url": image_url,
        "local_path": str(local_path) if local_path else "",
        "license": media.get("license") or record.get("license", ""),
        "rights_holder": media.get("rightsHolder") or record.get("rightsHolder", ""),
        "publisher": record.get("publisher", ""),
        "dataset_key": record.get("datasetKey", ""),
        "references": media.get("references") or record.get("references", ""),
        "decimal_latitude": record.get("decimalLatitude", ""),
        "decimal_longitude": record.get("decimalLongitude", ""),
        "event_date": record.get("eventDate", ""),
        "downloaded": downloaded,
        "skip_reason": skip_reason,
    }


def safe_download_image(url: str, path_without_suffix: Path, overwrite: bool = False) -> tuple[bool, Path | None, str]:
    if not url:
        return False, None, "missing_url"

    try:
        with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "coral-fish-pipeline/gbif-media"}) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type")
            ext = infer_extension(content_type, url)
            if ext is None:
                return False, None, f"non_image_content_type:{content_type or 'unknown'}"
            out_path = path_without_suffix.with_suffix(ext)
            if out_path.exists() and not overwrite:
                return True, out_path, "exists"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
            return True, out_path, ""
    except Exception as exc:
        return False, None, f"download_failed:{exc}"


def iter_media_candidates(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    candidates: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for record in records:
        for idx, media in enumerate(record.get("media") or []):
            if isinstance(media, dict):
                candidates.append((record, media, idx))
    return candidates


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def download_region(args: argparse.Namespace) -> None:
    species_list = load_region_species(Path(args.species_yaml), args.region)
    if args.limit_species is not None:
        species_list = species_list[: args.limit_species]

    out_root = Path(args.out) / args.region
    out_root.mkdir(parents=True, exist_ok=True)
    allowed = {normalize_license(item) for item in args.license}
    manifest_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    iterator = tqdm(species_list, desc=f"GBIF {args.region}") if tqdm else species_list

    for species in iterator:
        downloaded_count = 0
        skipped_count = 0
        slug = slug_species(species)
        species_dir = out_root / slug

        try:
            records = query_gbif_records(species, args.region, source_dataset=args.source_dataset)
        except Exception as exc:
            print(f"GBIF query failed for {species}: {exc}")
            count_rows.append({"species": species, "requested_max": args.max_per_species, "downloaded_count": 0, "skipped_count": 1})
            continue

        candidates = iter_media_candidates(records)
        if args.dry_run:
            print(f"{args.region}: {species}: {len(records)} records, {len(candidates)} media candidates")
            count_rows.append({"species": species, "requested_max": args.max_per_species, "downloaded_count": 0, "skipped_count": 0})
            continue

        for record, media, media_idx in candidates:
            if downloaded_count >= args.max_per_species:
                break
            image_url = media_url(media)
            license_value = str(media.get("license") or record.get("license") or "")
            if not is_allowed_license(license_value, allowed):
                skipped_count += 1
                manifest_rows.append(build_manifest_row(args.region, species, record, media, image_url, None, False, "license_not_allowed"))
                continue

            gbif_id = str(record.get("gbifID") or "unknown")
            target_base = species_dir / f"{slug}_{gbif_id}_{media_idx}"
            print(f"Downloading {args.region}: {species} {downloaded_count + 1}/{args.max_per_species}")
            ok, local_path, reason = safe_download_image(image_url, target_base, overwrite=args.overwrite)
            if ok:
                downloaded_count += 1
                manifest_rows.append(build_manifest_row(args.region, species, record, media, image_url, local_path, True, "" if reason != "exists" else "exists"))
            else:
                skipped_count += 1
                manifest_rows.append(build_manifest_row(args.region, species, record, media, image_url, None, False, reason))
            time.sleep(args.sleep)

        count_rows.append(
            {
                "species": species,
                "requested_max": args.max_per_species,
                "downloaded_count": downloaded_count,
                "skipped_count": skipped_count,
            }
        )

    if not args.dry_run:
        write_csv(out_root / "manifest.csv", manifest_rows, MANIFEST_FIELDS)
        write_csv(out_root / "species_counts.csv", count_rows, COUNTS_FIELDS)
        print(f"Wrote manifest: {out_root / 'manifest.csv'}")
        print(f"Wrote species counts: {out_root / 'species_counts.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GBIF occurrence media for region top-25 species.")
    parser.add_argument("--region", required=True, choices=["fiji", "gulf_of_thailand"])
    parser.add_argument("--species-yaml", default="resources/top25.yaml")
    parser.add_argument("--out", default="data/gbif_media")
    parser.add_argument("--max-per-species", type=int, default=25)
    parser.add_argument("--license", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--limit-species", type=int, default=None)
    parser.add_argument("--source-dataset", default=None)
    parser.add_argument("--quality", default="none")
    args = parser.parse_args()
    if args.license is None:
        args.license = DEFAULT_LICENSES
    download_region(args)


if __name__ == "__main__":
    main()
