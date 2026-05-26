from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from tempfile import TemporaryDirectory

from tqdm import tqdm

from coral_fish_pipeline.config import ensure_output_dirs
from coral_fish_pipeline.io.image_loader import list_images, load_image_rgb, image_id_from_path
from coral_fish_pipeline.io.metadata_writer import write_jsonl, write_json, write_csv
from coral_fish_pipeline.preprocessing.preprocess import apply_preprocessing
from coral_fish_pipeline.segmentation.mock_segmenter import MockSegmenter
from coral_fish_pipeline.segmentation.sam3_runner import SAM3Runner
from coral_fish_pipeline.segmentation.postprocess import postprocess_detections
from coral_fish_pipeline.cropping.cropper import create_crops
from coral_fish_pipeline.classification.region_loader import load_region_species
from coral_fish_pipeline.classification.bioclip_classifier import BioCLIPClassifier, sort_crops_by_species
from coral_fish_pipeline.visualization.overlays import save_overlays
from coral_fish_pipeline.visualization.contact_sheet import make_contact_sheet
from coral_fish_pipeline.models import Detection, CropRecord, ClassificationResult
from coral_fish_pipeline.evaluation.detection_eval import evaluate_detection_for_image, summarize_eval
from coral_fish_pipeline.io.yolo_loader import get_split_dirs, read_yolo_labels_for_image


def build_segmenter(backend: str, cfg: dict[str, Any]):
    if backend == "mock":
        return MockSegmenter(**cfg)
    if backend == "sam3":
        return SAM3Runner(**cfg)
    raise ValueError(f"Unknown segmentation backend: {backend}")


def make_summary(results: list[ClassificationResult], detections: list[Detection]) -> dict[str, Any]:
    accepted = sum(1 for d in detections if d.status == "accepted")
    rejected = sum(1 for d in detections if d.status == "rejected")
    counts = Counter(r.predicted_species if r.status != "uncertain" else f"Uncertain:{r.predicted_species}" for r in results)
    total_classified = sum(counts.values())
    species = []
    for name, count in counts.most_common():
        species.append({"species": name, "count": count, "percent": (100.0 * count / total_classified) if total_classified else 0.0})
    return {
        "total_detections": len(detections),
        "accepted_detections": accepted,
        "rejected_detections": rejected,
        "classified_crops": total_classified,
        "species_breakdown": species,
    }


def write_summary_text(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"Total detections: {summary['total_detections']}",
        f"Accepted detections: {summary['accepted_detections']}",
        f"Rejected detections: {summary['rejected_detections']}",
        f"Classified crops: {summary['classified_crops']}",
        "",
        "Species breakdown:",
    ]
    for row in summary["species_breakdown"]:
        lines.append(f"- {row['species']}: {row['count']} ({row['percent']:.1f}%)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _output_cfg(cfg: dict[str, Any]) -> dict[str, bool]:
    output = dict(cfg.get("output", {}))
    return {
        "save_crops": bool(output.get("save_crops", False)),
        "save_masks": bool(output.get("save_masks", False)),
        "save_overlays": bool(output.get("save_overlays", False)),
        "save_contact_sheet": bool(output.get("save_contact_sheet", False)),
    }


def _detection_dict(det: Detection, save_masks: bool) -> dict[str, Any]:
    data = det.to_dict()
    if not save_masks:
        data["mask_path"] = None
    return data


def _crop_dict(crop: CropRecord, save_crops: bool) -> dict[str, Any]:
    data = crop.to_dict()
    if not save_crops:
        data["raw_crop_path"] = None
        data["masked_crop_path"] = None
    return data


def _classification_rows(
    results: list[ClassificationResult],
    crops: list[CropRecord],
    save_crops: bool,
) -> list[dict[str, Any]]:
    crops_by_id = {c.crop_id: c for c in crops}
    rows = []
    for r in results:
        crop = crops_by_id.get(r.crop_id)
        row = {
            "crop_id": r.crop_id,
            "det_id": crop.det_id if crop else r.crop_id,
            "image_id": crop.image_id if crop else None,
            "bbox_xyxy": repr(crop.bbox_original) if crop else None,
            "crop_path": crop.raw_crop_path if crop and save_crops else None,
            "masked_crop_path": crop.masked_crop_path if crop and save_crops else None,
            "predicted_species": r.predicted_species,
            "confidence": r.confidence,
            "status": r.status,
            "region": r.region,
            "model_id": r.model_id,
            "top5": repr(r.top5),
        }
        rows.append(row)
    return rows


def _classification_dicts(
    results: list[ClassificationResult],
    crops: list[CropRecord],
    save_crops: bool,
) -> list[dict[str, Any]]:
    crops_by_id = {c.crop_id: c for c in crops}
    records = []
    for result in results:
        data = result.to_dict()
        crop = crops_by_id.get(result.crop_id)
        if crop:
            data.update(
                {
                    "det_id": crop.det_id,
                    "image_id": crop.image_id,
                    "bbox_xyxy": crop.bbox_original,
                    "crop_path": crop.raw_crop_path if save_crops else None,
                    "masked_crop_path": crop.masked_crop_path if save_crops else None,
                }
            )
        records.append(data)
    return records


def run_pipeline(
    input_path: str | Path,
    region: str,
    output_dir: str | Path,
    cfg: dict[str, Any],
    region_yaml: str | Path,
    limit: int | None = None,
    segmenter_backend: str | None = None,
    skip_classification: bool = False,
    bioclip_model: str | None = None,
) -> dict[str, Any]:
    output_cfg = _output_cfg(cfg)
    dirs = ensure_output_dirs(output_dir, create_artifact_dirs=False)
    images = list_images(input_path, cfg["input"].get("image_extensions"), limit=limit)
    if not images:
        raise FileNotFoundError(f"No images found in {input_path}")

    species = load_region_species(region_yaml, region)
    seg_cfg = dict(cfg.get("segmentation", {}))
    backend = segmenter_backend or seg_cfg.get("backend", "sam3")
    seg_cfg.pop("backend", None)
    segmenter = build_segmenter(backend, seg_cfg)

    all_detections: list[Detection] = []
    all_crops: list[CropRecord] = []
    overlay_paths: list[str] = []

    prep_cfg = dict(cfg.get("preprocessing", {}))
    with TemporaryDirectory(prefix="coral_fish_pipeline_") as tmp:
        tmp_root = Path(tmp)
        mask_root = dirs["root"] / "masks" if output_cfg["save_masks"] else tmp_root / "masks"
        crops_raw_dir = dirs["crops_raw"] if output_cfg["save_crops"] else tmp_root / "crops" / "raw"
        crops_masked_dir = dirs["crops_masked"] if output_cfg["save_crops"] else tmp_root / "crops" / "masked"

        for img_path in tqdm(images, desc="Segment/crop"):
            image_id = image_id_from_path(img_path)
            image = load_image_rgb(img_path)
            seg_image = image
            if prep_cfg.get("apply_to_segmentation", True):
                seg_image = apply_preprocessing(image, **prep_cfg)
            crop_image = image
            if prep_cfg.get("apply_to_classification", False):
                crop_image = apply_preprocessing(image, **prep_cfg)
            mask_dir = mask_root / image_id
            detections = segmenter.predict(seg_image, image_id=image_id, output_mask_dir=mask_dir, prompts=seg_cfg.get("prompts", []))
            detections = postprocess_detections(detections, image, cfg.get("postprocess", {}))
            all_detections.extend(detections)
            if output_cfg["save_overlays"]:
                paths = save_overlays(image, image_id, detections, dirs["overlays"])
                overlay_paths.append(paths["all"])
            crops = create_crops(
                crop_image,
                image_id,
                detections,
                crops_raw_dir,
                crops_masked_dir,
                **cfg.get("crop", {}),
            )
            all_crops.extend(crops)

        write_jsonl(dirs["metadata"] / "detections.jsonl", [_detection_dict(d, output_cfg["save_masks"]) for d in all_detections])
        write_jsonl(dirs["metadata"] / "crops.jsonl", [_crop_dict(c, output_cfg["save_crops"]) for c in all_crops])

        results: list[ClassificationResult] = []
        if not skip_classification and all_crops:
            class_cfg = dict(cfg.get("classification", {}))
            if bioclip_model:
                class_cfg["primary_model_id"] = bioclip_model
            classifier = BioCLIPClassifier(
                species=species,
                region=region,
                cache_dir=dirs["cache"],
                **class_cfg,
            )
            results = classifier.classify_crops(all_crops)
            if output_cfg["save_crops"]:
                sort_crops_by_species(all_crops, results, dirs["crops_by_species"])

        result_dicts = _classification_dicts(results, all_crops, output_cfg["save_crops"])
        write_jsonl(dirs["metadata"] / "classifications.jsonl", result_dicts)
        write_csv(dirs["metadata"] / "classifications.csv", _classification_rows(results, all_crops, output_cfg["save_crops"]))

        summary = make_summary(results, all_detections)
        write_json(dirs["metadata"] / "summary.json", summary)
        write_summary_text(dirs["metadata"] / "summary.txt", summary)
        if output_cfg["save_contact_sheet"]:
            try:
                make_contact_sheet([str(p) for p in images], overlay_paths, all_crops, results, dirs["root"] / "contact_sheet.jpg")
            except Exception as exc:
                print(f"Warning: could not create contact sheet: {exc}")
        return summary


def run_eval(
    dataset: str | Path,
    split: str,
    region: str,
    output_dir: str | Path,
    cfg: dict[str, Any],
    region_yaml: str | Path,
    limit: int | None = None,
    segmenter_backend: str | None = None,
    skip_classification: bool = True,
    bioclip_model: str | None = None,
) -> dict[str, Any]:
    img_dir, label_dir = get_split_dirs(dataset, split)
    summary = run_pipeline(
        input_path=img_dir,
        region=region,
        output_dir=output_dir,
        cfg=cfg,
        region_yaml=region_yaml,
        limit=limit,
        segmenter_backend=segmenter_backend,
        skip_classification=skip_classification,
        bioclip_model=bioclip_model,
    )
    dirs = ensure_output_dirs(output_dir, create_artifact_dirs=False)
    # Reload predictions grouped by image.
    detections: list[Detection] = []
    import json
    det_file = dirs["metadata"] / "detections.jsonl"
    for line in det_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            detections.append(Detection(**json.loads(line)))
    by_image: dict[str, list[Detection]] = defaultdict(list)
    for det in detections:
        by_image[det.image_id].append(det)

    image_paths = list_images(img_dir, cfg["input"].get("image_extensions"), limit=limit)
    image_metrics = []
    for img_path in image_paths:
        image_id = image_id_from_path(img_path)
        gt = read_yolo_labels_for_image(img_path, label_dir)
        m = evaluate_detection_for_image(by_image.get(image_id, []), [g.bbox_xyxy for g in gt], cfg.get("evaluation", {}).get("iou_threshold", 0.5))
        m["image_id"] = image_id
        image_metrics.append(m)
    metrics = summarize_eval(image_metrics)
    metrics["per_image"] = image_metrics
    write_json(dirs["metadata"] / "metrics.json", metrics)
    lines = [
        f"Images: {metrics['num_images']}",
        f"TP: {metrics['tp']}",
        f"FP: {metrics['fp']}",
        f"FN: {metrics['fn']}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1: {metrics['f1']:.4f}",
        f"AP50 proxy: {metrics['ap50']:.4f}",
    ]
    (dirs["metadata"] / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"pipeline_summary": summary, "metrics": metrics}
