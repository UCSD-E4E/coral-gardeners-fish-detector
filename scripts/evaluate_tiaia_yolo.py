from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from coral_fish_pipeline.classification.bioclip_classifier import BioCLIPClassifier
from coral_fish_pipeline.classification.region_loader import load_region_species
from coral_fish_pipeline.config import load_config
from coral_fish_pipeline.evaluation.metrics import match_predictions_to_ground_truth, precision_recall_f1
from coral_fish_pipeline.io.yolo_loader import get_split_dirs, read_yolo_labels_for_image
from coral_fish_pipeline.models import CropRecord, Detection
from coral_fish_pipeline.preprocessing.preprocess import apply_preprocessing
from coral_fish_pipeline.segmentation.postprocess import postprocess_detections
from coral_fish_pipeline.utils.paths import safe_name


DEFAULT_EVAL_CLASSES = {"brown_tang", "butterflyfish", "fish", "parrotfish", "surgeonfish"}
DEFAULT_IGNORE_CLASSES = {"cushion_starfish"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PER_IMAGE_FIELDS = ["image", "gt_count", "pred_count", "tp", "fp", "fn", "precision", "recall", "f1"]
MATCHED_CROP_FIELDS = [
    "image",
    "crop_id",
    "crop_path",
    "yolo_class",
    "matched_iou",
    "gt_box_xyxy",
    "pred_box_xyxy",
    "acceptable_species",
    "detection_score",
]
CLASSIFICATION_FIELDS = [
    "image",
    "crop_path",
    "yolo_class",
    "matched_iou",
    "predicted_species",
    "confidence",
    "top1_correct",
    "top5_correct",
    "acceptable_species",
    "top5",
    "model_id",
    "gt_box_xyxy",
    "pred_box_xyxy",
]


@dataclass
class GroundTruthBox:
    bbox_xyxy: list[float]
    class_id: int
    class_name: str


@dataclass
class ClassificationContext:
    classifier: BioCLIPClassifier
    class_map: dict[str, list[str]]
    region_species: list[str]
    output_dir: Path
    crops_by_prediction_dir: Path
    save_prediction_crops: bool = False


def load_class_names(dataset_root: Path) -> list[str]:
    data_yaml_path = dataset_root / "data.yaml"
    if not data_yaml_path.exists():
        data_yaml_path = dataset_root / "data.yml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Could not find data.yaml or data.yml in {dataset_root}")

    data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names, key=lambda k: int(k))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"Expected YOLO class names list or mapping in {data_yaml_path}")


def load_tiaia_class_map(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Tiaia class map not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    yolo_to_latin = data.get("yolo_to_latin")
    if not isinstance(yolo_to_latin, dict):
        raise ValueError(f"Expected yolo_to_latin mapping in {path}")
    return data


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def select_images(image_paths: list[Path], args: argparse.Namespace) -> tuple[list[Path], str]:
    total = len(image_paths)
    if args.shard_index is not None or args.num_shards is not None:
        if args.shard_index is None or args.num_shards is None:
            raise ValueError("--shard-index and --num-shards must be provided together")
        if args.num_shards <= 0:
            raise ValueError("--num-shards must be greater than zero")
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise ValueError("--shard-index must be in [0, num_shards)")
        selected = [p for idx, p in enumerate(image_paths) if idx % args.num_shards == args.shard_index]
        return selected, f"shard {args.shard_index}/{args.num_shards}"

    if args.start_index != 0 or args.end_index is not None:
        end = args.end_index if args.end_index is not None else total
        selected = image_paths[args.start_index:end]
        return selected, f"indices [{args.start_index}:{end}]"

    if args.limit is not None:
        return image_paths[: args.limit], f"limit {args.limit}"

    return image_paths, "all images"


def load_filtered_ground_truth(
    image_path: Path,
    labels_dir: Path,
    class_names: list[str],
    eval_classes: set[str],
    ignore_classes: set[str],
) -> list[GroundTruthBox]:
    boxes = []
    for box in read_yolo_labels_for_image(image_path, labels_dir):
        class_name = class_names[box.class_id] if 0 <= box.class_id < len(class_names) else str(box.class_id)
        if class_name in ignore_classes:
            continue
        if eval_classes and class_name not in eval_classes:
            continue
        boxes.append(GroundTruthBox(box.bbox_xyxy, box.class_id, class_name))
    return boxes


def padded_crop(image: Image.Image, box: list[float], padding_percent: float = 0.30) -> Image.Image:
    x1, y1, x2, y2 = map(float, box)
    width, height = image.size
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * padding_percent
    pad_y = bh * padding_percent
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    return image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))


def format_box(box: list[float]) -> str:
    return json.dumps([round(float(v), 2) for v in box])


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in {None, ""}:
        return []
    return json.loads(str(value))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill: tuple[int, int, int, int], font: Any) -> None:
    x, y = xy
    try:
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        draw.rectangle([left - 2, top - 1, right + 2, bottom + 1], fill=(0, 0, 0, 150))
    except Exception:
        pass
    draw.text((x, y), text, fill=fill, font=font)


def draw_eval_overlay(
    image: Image.Image,
    detections: list[Detection],
    gt_boxes: list[GroundTruthBox],
    matches: list[tuple[int, int, float]],
    false_positive_indices: list[int],
    false_negative_indices: list[int],
    output_path: Path,
    mode: str = "all",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    accepted = [d for d in detections if d.status == "accepted"]
    tp_pred = {pi: (gi, iou) for pi, gi, iou in matches}
    fp_set = set(false_positive_indices)
    fn_set = set(false_negative_indices)

    if mode in {"all", "false_negatives"}:
        for gi, gt in enumerate(gt_boxes):
            if mode == "false_negatives" and gi not in fn_set:
                continue
            x1, y1, x2, y2 = gt.bbox_xyxy
            is_fn = gi in fn_set
            color = (255, 165, 0, 235) if is_fn else (0, 128, 255, 220)
            fill = (255, 165, 0, 35) if is_fn else (0, 128, 255, 25)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.rectangle([x1, y1, x2, y2], fill=fill)
            label = f"FN {gt.class_name}" if is_fn else f"GT {gt.class_name}"
            draw_label(draw, (x1 + 2, max(0, y1 - 18)), label, color, font)

    if mode in {"all", "false_positives"}:
        for pi, det in enumerate(accepted):
            if mode == "false_positives" and pi not in fp_set:
                continue
            if mode == "all" and pi not in tp_pred and pi not in fp_set:
                continue
            x1, y1, x2, y2 = det.bbox_xyxy
            if pi in tp_pred:
                _, iou = tp_pred[pi]
                color = (0, 220, 0, 235)
                fill = (0, 220, 0, 35)
                label = f"TP {det.score:.2f} IoU {iou:.2f}"
            else:
                color = (255, 0, 0, 235)
                fill = (255, 0, 0, 35)
                label = f"FP {det.score:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.rectangle([x1, y1, x2, y2], fill=fill)
            draw_label(draw, (x1 + 2, min(out.height - 16, y2 + 2)), label, color, font)

    out.save(output_path, quality=95)


def acceptable_for_class(class_name: str, class_map: dict[str, list[str]] | None) -> list[str]:
    if class_map is None:
        return []
    return class_map.get(class_name, [])


def make_matched_crop_records(
    image_path: Path,
    image: Image.Image,
    accepted: list[Detection],
    gt_boxes: list[GroundTruthBox],
    matches: list[tuple[int, int, float]],
    crops_dir: Path,
    class_map: dict[str, list[str]] | None,
    save_crops: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    crops_dir.mkdir(parents=True, exist_ok=True)
    for match_idx, (pred_idx, gt_idx, matched_iou) in enumerate(matches):
        det = accepted[pred_idx]
        gt = gt_boxes[gt_idx]
        crop_id = f"{image_path.stem}_{match_idx:04d}_{det.det_id}"
        crop_path = crops_dir / f"{crop_id}.jpg"
        if save_crops and not crop_path.exists():
            padded_crop(image, det.bbox_xyxy).save(crop_path, quality=95)
        records.append(
            {
                "image": image_path.name,
                "crop_id": crop_id,
                "crop_path": str(crop_path),
                "yolo_class": gt.class_name,
                "matched_iou": matched_iou,
                "gt_box_xyxy": format_box(gt.bbox_xyxy),
                "pred_box_xyxy": format_box(det.bbox_xyxy),
                "acceptable_species": json.dumps(acceptable_for_class(gt.class_name, class_map)),
                "detection_score": det.score,
            }
        )
    return records


def checkpoint_path(checkpoints_dir: Path, image_path: Path) -> Path:
    return checkpoints_dir / f"{image_path.stem}.json"


def load_checkpoint(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "image": data["image"],
        "gt_count": data["gt_count"],
        "pred_count": data["pred_count"],
        "tp": data["tp"],
        "fp": data["fp"],
        "fn": data["fn"],
        "precision": data["precision"],
        "recall": data["recall"],
        "f1": data["f1"],
    }
    return row, list(data.get("matched_crop_records", []))


def write_checkpoint(path: Path, row: dict[str, Any], matched_crop_records: list[dict[str, Any]]) -> None:
    data = dict(row)
    data["matched_crop_records"] = matched_crop_records
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def add_row_to_totals(totals: dict[str, int], row: dict[str, Any]) -> None:
    totals["tp"] += int(row["tp"])
    totals["fp"] += int(row["fp"])
    totals["fn"] += int(row["fn"])
    totals["gt"] += int(row["gt_count"])
    totals["pred"] += int(row["pred_count"])


def release_sam3_memory(segmenter: Any) -> None:
    del segmenter
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    print("Released SAM3 GPU memory before BioCLIP classification.")


def is_prediction_acceptable(predicted_species: str, acceptable_species: list[str], region_species: list[str]) -> bool:
    if "*" in acceptable_species:
        return predicted_species in region_species
    return predicted_species in acceptable_species


def top5_has_acceptable(top5: list[tuple[str, float]], acceptable_species: list[str], region_species: list[str]) -> bool:
    return any(is_prediction_acceptable(species, acceptable_species, region_species) for species, _ in top5)


def build_classification_context(args: argparse.Namespace, cfg: dict[str, Any], output_dir: Path) -> ClassificationContext:
    class_output_dir = Path(args.classification_output_dir) if args.classification_output_dir else output_dir / "classification"
    crops_by_prediction_dir = class_output_dir / "crops_by_prediction"
    class_output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_matched_crops or args.save_crops:
        crops_by_prediction_dir.mkdir(parents=True, exist_ok=True)

    class_map_data = load_tiaia_class_map(Path(args.class_map))
    region_species = load_region_species("resources/top25.yaml", args.region)
    class_cfg = dict(cfg.get("classification", {}))
    classifier = BioCLIPClassifier(
        species=region_species,
        region=args.region,
        primary_model_id=args.bioclip_model,
        fallback_model_id=args.bioclip_model,
        device=class_cfg.get("device", "auto"),
        precision=class_cfg.get("precision", "fp16"),
        batch_size=class_cfg.get("batch_size", 1),
        unknown_threshold=0.0 if args.force_species_guess else class_cfg.get("unknown_threshold", 0.25),
        uncertain_margin=0.0 if args.force_species_guess else class_cfg.get("uncertain_margin", 0.08),
        cache_dir=class_output_dir / ".cache",
        cache_text_embeddings=class_cfg.get("cache_text_embeddings", True),
        classify_masked_crops=False,
        allow_original_bioclip=False,
    )
    return ClassificationContext(
        classifier=classifier,
        class_map={str(k): list(v) for k, v in class_map_data["yolo_to_latin"].items()},
        region_species=region_species,
        output_dir=class_output_dir,
        crops_by_prediction_dir=crops_by_prediction_dir,
        save_prediction_crops=bool(args.save_matched_crops or args.save_crops),
    )


def classify_matched_crops(rows: list[dict[str, Any]], context: ClassificationContext) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="Classify matched crops"):
        crop_path = Path(row["crop_path"])
        if not crop_path.exists():
            raise FileNotFoundError(f"Matched crop does not exist: {crop_path}")
        crop_record = CropRecord(
            crop_id=str(row.get("crop_id") or crop_path.stem),
            image_id=Path(str(row["image"])).stem,
            det_id=str(row.get("crop_id") or crop_path.stem),
            raw_crop_path=str(crop_path),
            masked_crop_path=None,
            bbox_original=parse_json_list(row.get("pred_box_xyxy")),
            bbox_padded=parse_json_list(row.get("pred_box_xyxy")),
        )
        result = context.classifier.classify_crop(crop_record)
        top5 = result.top5
        predicted_species = top5[0][0] if top5 else result.predicted_species
        confidence = float(top5[0][1]) if top5 else float(result.confidence)
        acceptable = parse_json_list(row.get("acceptable_species"))
        if not acceptable:
            acceptable = context.class_map.get(str(row["yolo_class"]), [])
        top1_correct = is_prediction_acceptable(predicted_species, acceptable, context.region_species) if acceptable else False
        top5_correct = top5_has_acceptable(top5, acceptable, context.region_species) if acceptable else False

        if context.save_prediction_crops:
            pred_dir = context.crops_by_prediction_dir / safe_name(predicted_species)
            pred_dir.mkdir(parents=True, exist_ok=True)
            target_path = pred_dir / crop_path.name
            if not target_path.exists():
                target_path.write_bytes(crop_path.read_bytes())

        out_rows.append(
            {
                "image": row["image"],
                "crop_path": str(crop_path),
                "yolo_class": row["yolo_class"],
                "matched_iou": row["matched_iou"],
                "predicted_species": predicted_species,
                "confidence": confidence,
                "top1_correct": top1_correct,
                "top5_correct": top5_correct,
                "acceptable_species": json.dumps(acceptable),
                "top5": json.dumps(top5),
                "model_id": result.model_id,
                "gt_box_xyxy": row["gt_box_xyxy"],
                "pred_box_xyxy": row["pred_box_xyxy"],
            }
        )
    return out_rows


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def summarize_classification(rows: list[dict[str, Any]], total_ground_truth: int | None, class_map: dict[str, list[str]]) -> dict[str, Any]:
    top1_correct = sum(1 for row in rows if bool_value(row["top1_correct"]))
    top5_correct = sum(1 for row in rows if bool_value(row["top5_correct"]))
    confidences = [float(row["confidence"]) for row in rows]
    unmapped = sorted({row["yolo_class"] for row in rows if row["yolo_class"] not in class_map})
    count = len(rows)
    metrics: dict[str, Any] = {
        "matched_detections_classified": count,
        "top1_correct": top1_correct,
        "top1_incorrect": count - top1_correct,
        "top1_accuracy_on_matched_detections": top1_correct / count if count else 0.0,
        "top5_correct": top5_correct,
        "top5_accuracy_on_matched_detections": top5_correct / count if count else 0.0,
        "unmapped_yolo_classes": unmapped,
        "classification_confidence_mean": mean(confidences) if confidences else 0.0,
        "classification_confidence_median": median(confidences) if confidences else 0.0,
        "end_to_end_correct_top1": top1_correct,
        "end_to_end_correct_top5": top5_correct,
    }
    if total_ground_truth is not None:
        metrics["end_to_end_accuracy_top1_over_gt"] = top1_correct / total_ground_truth if total_ground_truth else 0.0
        metrics["end_to_end_accuracy_top5_over_gt"] = top5_correct / total_ground_truth if total_ground_truth else 0.0
    else:
        metrics["end_to_end_accuracy_top1_over_gt"] = None
        metrics["end_to_end_accuracy_top5_over_gt"] = None
    return metrics


def write_classification_outputs(context: ClassificationContext, rows: list[dict[str, Any]], total_ground_truth: int | None) -> dict[str, Any]:
    write_csv(context.output_dir / "matched_classifications.csv", rows, CLASSIFICATION_FIELDS)
    metrics = summarize_classification(rows, total_ground_truth, context.class_map)
    (context.output_dir / "classification_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = [
        "Classification on matched detections",
        "====================================",
        f"Matched detections classified: {metrics['matched_detections_classified']}",
        f"Top1 correct: {metrics['top1_correct']}",
        f"Top1 incorrect: {metrics['top1_incorrect']}",
        f"Top1 classification accuracy: {metrics['top1_accuracy_on_matched_detections']:.4f}",
        f"Top5 correct: {metrics['top5_correct']}",
        f"Top5 classification accuracy: {metrics['top5_accuracy_on_matched_detections']:.4f}",
        f"End-to-end top1 accuracy over GT: {metrics['end_to_end_accuracy_top1_over_gt']}",
        f"End-to-end top5 accuracy over GT: {metrics['end_to_end_accuracy_top5_over_gt']}",
        f"Confidence mean: {metrics['classification_confidence_mean']:.4f}",
        f"Confidence median: {metrics['classification_confidence_median']:.4f}",
        f"Unmapped YOLO classes: {metrics['unmapped_yolo_classes']}",
    ]
    (context.output_dir / "classification_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def find_total_gt_for_classification(args: argparse.Namespace, output_dir: Path) -> int | None:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "total_ground_truth" in metrics:
            return int(metrics["total_ground_truth"])
    if args.matched_crops_csv:
        source_metrics = Path(args.matched_crops_csv).parents[1] / "metrics.json"
        if source_metrics.exists():
            metrics = json.loads(source_metrics.read_text(encoding="utf-8"))
            if "total_ground_truth" in metrics:
                return int(metrics["total_ground_truth"])
    return None


def write_detection_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_root: Path,
    eval_classes: set[str],
    ignore_classes: set[str],
    total_images: int,
    selected_images: list[Path],
    selection_info: str,
    rows: list[dict[str, Any]],
    matched_crop_rows: list[dict[str, Any]],
    totals: dict[str, int],
    classification_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = precision_recall_f1(totals["tp"], totals["fp"], totals["fn"])
    metrics: dict[str, Any] = {
        "dataset": str(dataset_root),
        "split": args.split,
        "prompt": args.prompt,
        "use_tiling": args.use_tiling,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "preprocess": args.preprocess,
        "eval_classes": sorted(eval_classes),
        "ignored_classes": sorted(ignore_classes),
        "iou_threshold": args.iou_threshold,
        "total_images_in_split": total_images,
        "images_evaluated": len(selected_images),
        "selection": selection_info,
        "total_ground_truth": totals["gt"],
        "total_predictions": totals["pred"],
        "true_positives": totals["tp"],
        "false_positives": totals["fp"],
        "false_negatives": totals["fn"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "f1": summary["f1"],
        "per_image": rows,
    }
    if classification_metrics is not None:
        metrics["classification"] = classification_metrics

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(output_dir / "per_image_metrics.csv", rows, PER_IMAGE_FIELDS)
    persisted_crop_rows = matched_crop_rows
    if not (args.save_crops or args.save_matched_crops):
        persisted_crop_rows = [{**row, "crop_path": ""} for row in matched_crop_rows]
    write_csv(output_dir / "classification" / "matched_crops.csv", persisted_crop_rows, MATCHED_CROP_FIELDS)

    metrics_lines = [
        "Detection / Segmentation",
        "========================",
        f"Dataset: {dataset_root}",
        f"Split: {args.split}",
        f"Prompt: {args.prompt}",
        f"Eval classes: {sorted(eval_classes)}",
        f"Ignored classes: {sorted(ignore_classes)}",
        f"IoU threshold: {args.iou_threshold}",
        f"Total images in split: {total_images}",
        f"Selected images for this run: {len(selected_images)}",
        f"Selection: {selection_info}",
        "",
        f"Images evaluated: {len(selected_images)}",
        f"Ground-truth boxes: {totals['gt']}",
        f"Predicted boxes: {totals['pred']}",
        f"TP: {totals['tp']}",
        f"FP: {totals['fp']}",
        f"FN: {totals['fn']}",
        f"Precision: {summary['precision']:.4f}",
        f"Recall: {summary['recall']:.4f}",
        f"F1: {summary['f1']:.4f}",
    ]
    if classification_metrics is not None:
        metrics_lines.extend(
            [
                "",
                "Classification on matched detections",
                "====================================",
                f"Matched detections classified: {classification_metrics['matched_detections_classified']}",
                f"Top1 classification accuracy: {classification_metrics['top1_accuracy_on_matched_detections']:.4f}",
                f"Top5 classification accuracy: {classification_metrics['top5_accuracy_on_matched_detections']:.4f}",
                f"End-to-end top1 accuracy over GT: {classification_metrics['end_to_end_accuracy_top1_over_gt']}",
                f"End-to-end top5 accuracy over GT: {classification_metrics['end_to_end_accuracy_top5_over_gt']}",
            ]
        )
    (output_dir / "metrics.txt").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    print("\n".join(metrics_lines))
    print(f"\nMetrics written to: {output_dir / 'metrics.txt'}")
    return metrics


def run_classification_only(args: argparse.Namespace, cfg: dict[str, Any], output_dir: Path) -> None:
    matched_csv = Path(args.matched_crops_csv or args.classify_existing_crops)
    if matched_csv.is_dir():
        if matched_csv.name == "crops":
            matched_csv = matched_csv.parent / "matched_crops.csv"
        else:
            matched_csv = matched_csv / "matched_crops.csv"
    if not matched_csv.exists():
        raise FileNotFoundError(f"Matched crops CSV not found: {matched_csv}")
    context = build_classification_context(args, cfg, output_dir)
    matched_rows = read_csv_rows(matched_csv)
    class_rows = classify_matched_crops(matched_rows, context)
    metrics = write_classification_outputs(context, class_rows, find_total_gt_for_classification(args, output_dir))
    print("Classification on matched detections")
    print("====================================")
    print(f"Matched detections classified: {metrics['matched_detections_classified']}")
    print(f"Top1 classification accuracy: {metrics['top1_accuracy_on_matched_detections']:.4f}")
    print(f"Top5 classification accuracy: {metrics['top5_accuracy_on_matched_detections']:.4f}")
    print(f"End-to-end top1 accuracy over GT: {metrics['end_to_end_accuracy_top1_over_gt']}")
    print(f"End-to-end top5 accuracy over GT: {metrics['end_to_end_accuracy_top5_over_gt']}")


def run_detection(args: argparse.Namespace, cfg: dict[str, Any], output_dir: Path) -> None:
    from coral_fish_pipeline.segmentation.sam3_runner import SAM3Runner

    dataset_root = Path(args.dataset)
    overlays_dir = output_dir / "overlays"
    fp_dir = output_dir / "errors" / "false_positives"
    fn_dir = output_dir / "errors" / "false_negatives"
    checkpoints_dir = output_dir / "checkpoints"
    for directory in [output_dir, checkpoints_dir, output_dir / "classification"]:
        directory.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        for directory in [overlays_dir, fp_dir, fn_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    class_map = None
    if args.class_map:
        class_map = {str(k): list(v) for k, v in load_tiaia_class_map(Path(args.class_map))["yolo_to_latin"].items()}

    class_names = load_class_names(dataset_root)
    eval_classes = set(args.eval_class) if args.eval_class else set(DEFAULT_EVAL_CLASSES)
    ignore_classes = set(args.ignore_class) if args.ignore_class else set(DEFAULT_IGNORE_CLASSES)
    images_dir, labels_dir = get_split_dirs(dataset_root, args.split)
    all_images = list_images(images_dir)
    selected_images, selection_info = select_images(all_images, args)
    if not selected_images:
        raise FileNotFoundError(f"No selected images found in {images_dir}")

    print(f"Total images in dataset split: {len(all_images)}")
    print(f"Selected images for this run: {len(selected_images)}")
    print(f"Selection: {selection_info}")

    seg_cfg = cfg.get("segmentation", {})
    segmenter = SAM3Runner(
        min_confidence=seg_cfg.get("min_confidence", 0.25),
        max_detections_per_image=seg_cfg.get("max_detections_per_image", 100),
        nms_iou_threshold=seg_cfg.get("nms_iou_threshold", 0.55),
        use_tiling=args.use_tiling,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )
    post_cfg = cfg.get("postprocess", {})
    per_image_rows: list[dict[str, Any]] = []
    matched_crop_rows: list[dict[str, Any]] = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}

    with TemporaryDirectory(prefix="coral_fish_eval_") as tmp:
      tmp_root = Path(tmp)
      masks_dir = output_dir / "masks" if args.save_masks else tmp_root / "masks"
      crops_dir = output_dir / "classification" / "crops" if (args.save_crops or args.save_matched_crops) else tmp_root / "classification" / "crops"
      for image_path in tqdm(selected_images, desc="Evaluate SAM3"):
        ckpt = checkpoint_path(checkpoints_dir, image_path)
        if (args.resume or args.skip_existing) and ckpt.exists():
            row, crop_records = load_checkpoint(ckpt)
            per_image_rows.append(row)
            matched_crop_rows.extend(crop_records)
            add_row_to_totals(totals, row)
            continue

        image = Image.open(image_path).convert("RGB")
        seg_image = image
        if args.preprocess != "none":
            seg_image = apply_preprocessing(
                image,
                enabled=True,
                method=args.preprocess,
                apply_to_segmentation=True,
                apply_to_classification=False,
            )
        gt_boxes = load_filtered_ground_truth(image_path, labels_dir, class_names, eval_classes, ignore_classes)
        detections = segmenter.predict(image=seg_image, image_id=image_path.stem, output_mask_dir=masks_dir / image_path.stem, prompts=[args.prompt])
        detections = postprocess_detections(detections, image, post_cfg)
        accepted = [d for d in detections if d.status == "accepted"]
        pred_boxes = [d.bbox_xyxy for d in accepted]
        gt_xyxy = [gt.bbox_xyxy for gt in gt_boxes]
        matches, fp_idx, fn_idx = match_predictions_to_ground_truth(pred_boxes, gt_xyxy, args.iou_threshold)
        image_metrics = precision_recall_f1(len(matches), len(fp_idx), len(fn_idx))
        row = {
            "image": image_path.name,
            "gt_count": len(gt_boxes),
            "pred_count": len(pred_boxes),
            "tp": len(matches),
            "fp": len(fp_idx),
            "fn": len(fn_idx),
            "precision": image_metrics["precision"],
            "recall": image_metrics["recall"],
            "f1": image_metrics["f1"],
        }
        crop_records = make_matched_crop_records(
            image_path,
            image,
            accepted,
            gt_boxes,
            matches,
            crops_dir,
            class_map,
            args.save_crops or args.save_matched_crops or args.eval_classification,
        )
        per_image_rows.append(row)
        matched_crop_rows.extend(crop_records)
        add_row_to_totals(totals, row)

        if args.save_overlays:
            draw_eval_overlay(image, detections, gt_boxes, matches, fp_idx, fn_idx, overlays_dir / f"{image_path.stem}_overlay.jpg")
        if args.save_overlays and fp_idx:
            draw_eval_overlay(image, detections, gt_boxes, matches, fp_idx, fn_idx, fp_dir / f"{image_path.stem}_false_positives.jpg", mode="false_positives")
        if args.save_overlays and fn_idx:
            draw_eval_overlay(image, detections, gt_boxes, matches, fp_idx, fn_idx, fn_dir / f"{image_path.stem}_false_negatives.jpg", mode="false_negatives")
        write_checkpoint(ckpt, row, crop_records)

      classification_metrics = None
      if args.eval_classification and not args.detect_only:
          release_sam3_memory(segmenter)
          context = build_classification_context(args, cfg, output_dir)
          class_rows = classify_matched_crops(matched_crop_rows, context)
          classification_metrics = write_classification_outputs(context, class_rows, totals["gt"])

      write_detection_outputs(
          output_dir=output_dir,
          args=args,
          dataset_root=dataset_root,
          eval_classes=eval_classes,
          ignore_classes=ignore_classes,
          total_images=len(all_images),
          selected_images=selected_images,
          selection_info=selection_info,
          rows=per_image_rows,
          matched_crop_rows=matched_crop_rows,
          totals=totals,
          classification_metrics=classification_metrics,
      )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAM3 detection and optional BioCLIP classification on Tiaia YOLO.")
    parser.add_argument("--dataset", required=False, help="Path to YOLOv8 dataset root")
    parser.add_argument("--split", default="train", help="YOLO split name: train, valid, val, or test")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", default="configs/default.yaml", help="Project config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to evaluate")
    parser.add_argument("--prompt", default="fish", help="SAM3 text prompt")
    parser.add_argument("--use-tiling", action="store_true", help="Run SAM3 on full image plus overlapping tiles")
    parser.add_argument("--tile-size", type=int, default=768, help="SAM3 square tile size")
    parser.add_argument("--tile-overlap", type=float, default=0.25, help="SAM3 tile overlap fraction")
    parser.add_argument("--preprocess", choices=["none", "clahe_luminance"], default="none", help="Optional preprocessing for SAM3 detection only")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for TP matching")
    parser.add_argument("--eval-class", action="append", default=None, help="YOLO class to evaluate. Repeatable.")
    parser.add_argument("--ignore-class", action="append", default=None, help="YOLO class to ignore. Repeatable.")
    parser.add_argument("--eval-classification", action="store_true", help="Classify matched true-positive detections with BioCLIP")
    parser.add_argument("--region", default="moorea", help="Region key in resources/top25.yaml for BioCLIP candidates")
    parser.add_argument("--class-map", default="resources/tiaia_class_map.yaml", help="Tiaia YOLO-to-Latin class map")
    parser.add_argument("--bioclip-model", default="hf-hub:imageomics/bioclip-2", help="BioCLIP 2/2.5 model id")
    parser.add_argument("--classification-output-dir", default=None, help="Classification output directory")
    parser.add_argument("--force-species-guess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--save-matched-crops", action="store_true")
    parser.add_argument("--save-debug-artifacts", action="store_true")
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--classify-existing-crops", default=None, help="Path to matched_crops.csv or a directory containing it")
    parser.add_argument("--matched-crops-csv", default=None, help="Path to existing matched_crops.csv for classification-only mode")
    parser.add_argument("--merge-shards", default=None, help="Parent directory containing eval shard outputs to merge")
    args = parser.parse_args()
    if args.save_debug_artifacts:
        args.save_crops = True
        args.save_masks = True
        args.save_overlays = True
        args.save_matched_crops = True

    if args.merge_shards:
        from merge_tiaia_eval_shards import merge

        merge([Path(args.merge_shards)], Path(args.output))
        return

    cfg = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.matched_crops_csv or args.classify_existing_crops:
        if not args.eval_classification:
            raise ValueError("Classification-only mode requires --eval-classification")
        run_classification_only(args, cfg, output_dir)
        return
    if not args.dataset:
        raise ValueError("--dataset is required unless --matched-crops-csv or --classify-existing-crops is used")
    run_detection(args, cfg, output_dir)


if __name__ == "__main__":
    main()
