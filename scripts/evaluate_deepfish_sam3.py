from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw
from tqdm import tqdm

from coral_fish_pipeline.preprocessing.preprocess import apply_preprocessing
from coral_fish_pipeline.segmentation.sam3_runner import SAM3Runner


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class GTBox:
    xyxy: list[float]
    cls_id: int | None = None
    cls_name: str | None = None


def load_class_names(dataset_root: Path) -> list[str]:
    candidates = [
        dataset_root / "classes.txt",
        dataset_root / "classes",
        dataset_root / "data.yaml",
        dataset_root / "data.yml",
    ]

    for p in candidates:
        if not p.exists():
            continue

        if p.suffix in {".yaml", ".yml"}:
            try:
                import yaml

                data = yaml.safe_load(p.read_text())
                names = data.get("names", [])
                if isinstance(names, dict):
                    return [names[k] for k in sorted(names)]
                return list(names)
            except Exception:
                continue

        lines = [line.strip() for line in p.read_text().splitlines() if line.strip()]
        if lines:
            return lines

    return []


def yolo_line_to_xyxy(line: str, w: int, h: int, class_names: list[str]) -> GTBox | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        cls_id = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])
    except ValueError:
        return None

    # YOLO normalized xywh.
    x1 = (xc - bw / 2.0) * w
    y1 = (yc - bh / 2.0) * h
    x2 = (xc + bw / 2.0) * w
    y2 = (yc + bh / 2.0) * h

    cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else None
    return GTBox([x1, y1, x2, y2], cls_id=cls_id, cls_name=cls_name)


def load_yolo_boxes(label_path: Path, image_w: int, image_h: int, class_names: list[str]) -> list[GTBox]:
    if not label_path.exists():
        return []

    boxes: list[GTBox] = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        box = yolo_line_to_xyxy(line, image_w, image_h, class_names)
        if box is not None:
            boxes.append(box)

    return boxes


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def greedy_match(pred_boxes: list[list[float]], gt_boxes: list[GTBox], iou_threshold: float):
    matches: list[tuple[int, int, float]] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()

    candidates = []
    for pi, pred in enumerate(pred_boxes):
        for gi, gt in enumerate(gt_boxes):
            candidates.append((box_iou(pred, gt.xyxy), pi, gi))

    candidates.sort(reverse=True, key=lambda x: x[0])

    for iou, pi, gi in candidates:
        if iou < iou_threshold:
            break
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, iou))

    fp = [i for i in range(len(pred_boxes)) if i not in used_pred]
    fn = [i for i in range(len(gt_boxes)) if i not in used_gt]

    return matches, fp, fn


def find_images(dataset_root: Path, split: str, include_negative: bool) -> list[Path]:
    all_images = sorted(p for p in dataset_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

    if split != "all":
        split = split.lower()
        all_images = [
            p for p in all_images
            if any(part.lower() == split for part in p.parts)
        ]

    if not include_negative:
        all_images = [
            p for p in all_images
            if "negative" not in str(p).lower()
        ]

    return all_images


def draw_overlay(
    image: Image.Image,
    gt_boxes: list[GTBox],
    pred_boxes: list[list[float]],
    matches: list[tuple[int, int, float]],
    fp_indices: list[int],
    fn_indices: list[int],
    out_path: Path,
) -> None:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    matched_pred = {pi for pi, _, _ in matches}
    matched_gt = {gi for _, gi, _ in matches}

    # Ground truth boxes: blue unless missed, then red.
    for gi, gt in enumerate(gt_boxes):
        x1, y1, x2, y2 = gt.xyxy
        color = "red" if gi in fn_indices else "blue"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"GT {gt.cls_name or gt.cls_id}"
        draw.text((x1, max(0, y1 - 12)), label, fill=color)

    # Predicted boxes: green if matched, orange if false positive.
    for pi, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = box
        color = "green" if pi in matched_pred else "orange"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1), f"P{pi}", fill=color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to data/Deepfish")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "all"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="fish")
    parser.add_argument("--use-tiling", action="store_true", help="Run SAM3 on full image plus overlapping tiles")
    parser.add_argument("--tile-size", type=int, default=768, help="SAM3 square tile size")
    parser.add_argument("--tile-overlap", type=float, default=0.25, help="SAM3 tile overlap fraction")
    parser.add_argument("--preprocess", choices=["none", "clahe_luminance"], default="none", help="Optional preprocessing for SAM3 detection only")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.55)
    parser.add_argument("--include-negative", action="store_true")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlays_dir = output_dir / "overlays"
    metadata_dir = output_dir / "metadata"
    errors_dir = output_dir / "errors"

    for d in [metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        for d in [overlays_dir, errors_dir / "false_positives", errors_dir / "false_negatives"]:
            d.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(dataset_root)
    images = find_images(dataset_root, split=args.split, include_negative=args.include_negative)

    end = args.end_index if args.end_index is not None else len(images)
    images = images[args.start_index:end]

    if args.limit is not None:
        images = images[: args.limit]

    print(f"Dataset: {dataset_root}")
    print(f"Split: {args.split}")
    print(f"Images selected: {len(images)}")
    print(f"Class names loaded: {len(class_names)}")
    if class_names:
        print(f"First classes: {class_names[:10]}")

    segmenter = SAM3Runner(
        min_confidence=args.min_confidence,
        max_detections_per_image=args.max_detections,
        nms_iou_threshold=args.nms_iou_threshold,
        use_tiling=args.use_tiling,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )

    rows = []
    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    detections_jsonl = metadata_dir / "detections.jsonl"

    with TemporaryDirectory(prefix="coral_fish_eval_") as tmp, detections_jsonl.open("w") as det_f:
        masks_dir = output_dir / "masks" if args.save_masks else Path(tmp) / "masks"
        for image_path in tqdm(images, desc="Evaluate Deepfish SAM3"):
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
            image_id = image_path.stem

            label_path = image_path.with_suffix(".txt")
            gt_boxes = load_yolo_boxes(label_path, image.width, image.height, class_names)

            detections = segmenter.predict(
                image=seg_image,
                image_id=image_id,
                output_mask_dir=masks_dir / image_id,
                prompts=[args.prompt],
            )

            accepted = [d for d in detections if getattr(d, "status", "accepted") == "accepted"]
            pred_boxes = [list(map(float, d.bbox_xyxy)) for d in accepted]

            matches, fp_indices, fn_indices = greedy_match(
                pred_boxes=pred_boxes,
                gt_boxes=gt_boxes,
                iou_threshold=args.iou_threshold,
            )

            tp = len(matches)
            fp = len(fp_indices)
            fn = len(fn_indices)

            total_gt += len(gt_boxes)
            total_pred += len(pred_boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

            rows.append(
                {
                    "image": str(image_path),
                    "label": str(label_path),
                    "gt_count": len(gt_boxes),
                    "pred_count": len(pred_boxes),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

            for d in accepted:
                det_f.write(json.dumps({
                    "image": str(image_path),
                    "det_id": d.det_id,
                    "bbox_xyxy": d.bbox_xyxy,
                    "score": getattr(d, "score", None),
                    "prompt": getattr(d, "prompt", None),
                    "mask_path": getattr(d, "mask_path", None) if args.save_masks else None,
                }) + "\n")

            if args.save_overlays:
                draw_overlay(
                    image=image,
                    gt_boxes=gt_boxes,
                    pred_boxes=pred_boxes,
                    matches=matches,
                    fp_indices=fp_indices,
                    fn_indices=fn_indices,
                    out_path=overlays_dir / f"{image_id}_overlay.jpg",
                )

            if args.save_overlays and fp > 0:
                draw_overlay(
                    image=image,
                    gt_boxes=gt_boxes,
                    pred_boxes=[pred_boxes[i] for i in fp_indices],
                    matches=[],
                    fp_indices=list(range(len(fp_indices))),
                    fn_indices=[],
                    out_path=errors_dir / "false_positives" / f"{image_id}_fp.jpg",
                )

            if args.save_overlays and fn > 0:
                draw_overlay(
                    image=image,
                    gt_boxes=[gt_boxes[i] for i in fn_indices],
                    pred_boxes=[],
                    matches=[],
                    fp_indices=[],
                    fn_indices=list(range(len(fn_indices))),
                    out_path=errors_dir / "false_negatives" / f"{image_id}_fn.jpg",
                )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    metrics = {
        "dataset": str(dataset_root),
        "split": args.split,
        "prompt": args.prompt,
        "images_evaluated": len(images),
        "ground_truth_boxes": total_gt,
        "predicted_boxes": total_pred,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_threshold": args.iou_threshold,
        "min_confidence": args.min_confidence,
        "use_tiling": args.use_tiling,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "nms_iou_threshold": args.nms_iou_threshold,
        "preprocess": args.preprocess,
    }

    with (metadata_dir / "per_image_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    (metadata_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    summary = [
        "Deepfish SAM3 Evaluation",
        "========================",
        f"Dataset: {dataset_root}",
        f"Split: {args.split}",
        f"Prompt: {args.prompt}",
        f"Images evaluated: {len(images)}",
        "",
        f"Ground-truth boxes: {total_gt}",
        f"Predicted boxes: {total_pred}",
        f"TP: {total_tp}",
        f"FP: {total_fp}",
        f"FN: {total_fn}",
        "",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"F1: {f1:.4f}",
        "",
        f"Per-image metrics: {metadata_dir / 'per_image_metrics.csv'}",
        f"Overlays: {overlays_dir if args.save_overlays else 'disabled'}",
    ]

    (metadata_dir / "metrics.txt").write_text("\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
