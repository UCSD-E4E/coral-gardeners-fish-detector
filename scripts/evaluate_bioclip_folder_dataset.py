from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image

from coral_fish_pipeline.classification.bioclip_classifier import BioCLIPClassifier
from coral_fish_pipeline.classification.region_loader import load_region_species
from coral_fish_pipeline.models import CropRecord


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PREDICTION_FIELDS = [
    "image_path",
    "true_species",
    "predicted_species",
    "confidence",
    "top1_correct",
    "top5_correct",
    "top5",
    "model_id",
    "skip_reason",
]
PER_SPECIES_FIELDS = ["species", "images", "top1_correct", "top5_correct", "top1_accuracy", "top5_accuracy"]


def unslug_species(slug: str) -> str:
    return " ".join(slug.strip().split("_"))


def discover_folder_images(dataset: str | Path, candidate_species: list[str], limit: int | None = None) -> list[tuple[Path, str]]:
    dataset_path = Path(dataset)
    candidate_set = set(candidate_species)
    rows: list[tuple[Path, str]] = []
    for folder in sorted(p for p in dataset_path.iterdir() if p.is_dir()):
        species = unslug_species(folder.name)
        if species not in candidate_set:
            continue
        for image_path in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            rows.append((image_path, species))
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def topk_correct(top5: list[tuple[str, float]], true_species: str, k: int) -> bool:
    return any(species == true_species for species, _ in top5[:k])


def calculate_metrics(rows: list[dict[str, Any]], candidate_species_count: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluated = [row for row in rows if not row.get("skip_reason")]
    confidences = [float(row["confidence"]) for row in evaluated]
    top1_correct = sum(1 for row in evaluated if bool(row["top1_correct"]))
    top5_correct = sum(1 for row in evaluated if bool(row["top5_correct"]))
    species_names = sorted({str(row["true_species"]) for row in evaluated})

    metrics = {
        "images_evaluated": len(evaluated),
        "candidate_species_count": candidate_species_count,
        "evaluated_species_count": len(species_names),
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "top1_accuracy": top1_correct / len(evaluated) if evaluated else 0.0,
        "top5_accuracy": top5_correct / len(evaluated) if evaluated else 0.0,
        "confidence_mean": mean(confidences) if confidences else 0.0,
        "confidence_median": median(confidences) if confidences else 0.0,
    }

    per_species: list[dict[str, Any]] = []
    for species in species_names:
        species_rows = [row for row in evaluated if row["true_species"] == species]
        s_top1 = sum(1 for row in species_rows if bool(row["top1_correct"]))
        s_top5 = sum(1 for row in species_rows if bool(row["top5_correct"]))
        per_species.append(
            {
                "species": species,
                "images": len(species_rows),
                "top1_correct": s_top1,
                "top5_correct": s_top5,
                "top1_accuracy": s_top1 / len(species_rows) if species_rows else 0.0,
                "top5_accuracy": s_top5 / len(species_rows) if species_rows else 0.0,
            }
        )
    return metrics, per_species


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.allow_original_bioclip:
        raise ValueError("Original imageomics/bioclip is not allowed for this evaluator. Use BioCLIP 2.5 or BioCLIP 2.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    species = load_region_species(args.species_yaml, args.region)
    image_rows = discover_folder_images(args.dataset, species, limit=args.limit)
    classifier = BioCLIPClassifier(
        species=species,
        region=args.region,
        primary_model_id=args.bioclip_model,
        fallback_model_id=args.bioclip_model,
        batch_size=args.batch_size,
        unknown_threshold=0.0,
        uncertain_margin=0.0,
        cache_dir=output_dir / ".cache",
        cache_text_embeddings=True,
        classify_masked_crops=False,
        allow_original_bioclip=False,
    )

    predictions: list[dict[str, Any]] = []
    for idx, (image_path, true_species) in enumerate(image_rows):
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as exc:
            predictions.append(
                {
                    "image_path": str(image_path),
                    "true_species": true_species,
                    "predicted_species": "",
                    "confidence": "",
                    "top1_correct": False,
                    "top5_correct": False,
                    "top5": "[]",
                    "model_id": "",
                    "skip_reason": f"image_open_failed:{exc}",
                }
            )
            continue

        crop = CropRecord(
            crop_id=f"folder_{idx:06d}",
            image_id=image_path.stem,
            det_id=f"folder_{idx:06d}",
            raw_crop_path=str(image_path),
            masked_crop_path=None,
            bbox_original=[],
            bbox_padded=[],
        )
        result = classifier.classify_crop(crop)
        top5 = result.top5[: args.top_k]
        predicted = top5[0][0] if top5 else result.predicted_species
        predictions.append(
            {
                "image_path": str(image_path),
                "true_species": true_species,
                "predicted_species": predicted,
                "confidence": float(top5[0][1]) if top5 else float(result.confidence),
                "top1_correct": predicted == true_species,
                "top5_correct": topk_correct(top5, true_species, args.top_k),
                "top5": json.dumps(top5),
                "model_id": result.model_id,
                "skip_reason": "",
            }
        )

    metrics, per_species = calculate_metrics(predictions, candidate_species_count=len(species))
    write_csv(output_dir / "predictions.csv", predictions, PREDICTION_FIELDS)
    write_csv(output_dir / "per_species_metrics.csv", per_species, PER_SPECIES_FIELDS)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        "BioCLIP Folder Dataset Evaluation",
        "=================================",
        f"Dataset: {args.dataset}",
        f"Region: {args.region}",
        f"Model: {args.bioclip_model}",
        f"Candidate species: {metrics['candidate_species_count']}",
        f"Evaluated species: {metrics['evaluated_species_count']}",
        f"Images evaluated: {metrics['images_evaluated']}",
        f"Top1 correct: {metrics['top1_correct']}",
        f"Top5 correct: {metrics['top5_correct']}",
        f"Top1 accuracy: {metrics['top1_accuracy']:.4f}",
        f"Top5 accuracy: {metrics['top5_accuracy']:.4f}",
        f"Confidence mean: {metrics['confidence_mean']:.4f}",
        f"Confidence median: {metrics['confidence_median']:.4f}",
    ]
    (output_dir / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BioCLIP on a species-folder image dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--species-yaml", default="resources/top25.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bioclip-model", default="hf-hub:imageomics/bioclip-2.5-vith14")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-original-bioclip", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
