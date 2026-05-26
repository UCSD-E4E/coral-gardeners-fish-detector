from __future__ import annotations

import argparse
from pathlib import Path

from coral_fish_pipeline.config import load_config
from coral_fish_pipeline.pipeline import run_pipeline, run_eval


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--region", required=True, help="Region key from resources/top25.yaml, e.g. moorea, fiji")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", default=None, help="Optional YAML config override")
    parser.add_argument("--region-yaml", default="resources/top25.yaml", help="Path to top25 region YAML")
    parser.add_argument("--segmenter", choices=["sam3", "mock"], default=None, help="Override segmentation backend")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    parser.add_argument("--skip-classification", action="store_true", help="Skip BioCLIP classification")
    parser.add_argument("--bioclip-model", default=None, choices=["hf-hub:imageomics/bioclip-2.5-vith14", "hf-hub:imageomics/bioclip-2"], help="Override primary BioCLIP model")
    parser.add_argument("--use-tiling", action="store_true", help="Run SAM3 on the full image plus overlapping tiles")
    parser.add_argument("--tile-size", type=int, default=None, help="SAM3 tile size in pixels")
    parser.add_argument("--tile-overlap", type=float, default=None, help="SAM3 tile overlap fraction")
    parser.add_argument("--preprocess", choices=["none", "clahe_luminance"], default=None, help="Preprocess image for SAM3 detection")
    parser.add_argument("--save-crops", action="store_true", help="Persist crop images and crops_by_species")
    parser.add_argument("--save-masks", action="store_true", help="Persist segmentation masks")
    parser.add_argument("--save-overlays", action="store_true", help="Persist detection overlays")
    parser.add_argument("--save-contact-sheet", action="store_true", help="Persist contact_sheet.jpg")
    parser.add_argument("--save-debug-artifacts", action="store_true", help="Persist crops, masks, overlays, and contact sheet")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="coral-fish-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run full pipeline on image or folder")
    p_run.add_argument("--input", required=True, help="Image file or folder")
    add_common_args(p_run)

    p_smoke = sub.add_parser("smoke", help="Run a small debug pipeline on a few images")
    p_smoke.add_argument("--input", required=True, help="Image file or folder")
    add_common_args(p_smoke)
    p_smoke.set_defaults(limit_default=5)

    p_eval = sub.add_parser("eval", help="Evaluate on YOLO-labeled dataset")
    p_eval.add_argument("--dataset", required=True, help="YOLO dataset root")
    p_eval.add_argument("--split", default="test", help="train/val/test")
    add_common_args(p_eval)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if getattr(args, "use_tiling", False):
        cfg.setdefault("segmentation", {})["use_tiling"] = True
    if getattr(args, "tile_size", None) is not None:
        cfg.setdefault("segmentation", {})["tile_size"] = args.tile_size
    if getattr(args, "tile_overlap", None) is not None:
        cfg.setdefault("segmentation", {})["tile_overlap"] = args.tile_overlap
    if getattr(args, "preprocess", None) is not None:
        prep = cfg.setdefault("preprocessing", {})
        prep["enabled"] = args.preprocess != "none"
        prep["method"] = args.preprocess
        prep.setdefault("apply_to_segmentation", True)
        prep.setdefault("apply_to_classification", False)
    output = cfg.setdefault("output", {})
    if getattr(args, "save_debug_artifacts", False):
        output["save_crops"] = True
        output["save_masks"] = True
        output["save_overlays"] = True
        output["save_contact_sheet"] = True
    else:
        for name in ["save_crops", "save_masks", "save_overlays", "save_contact_sheet"]:
            if getattr(args, name, False):
                output[name] = True
    region_yaml = Path(args.region_yaml)
    if not region_yaml.exists():
        # Running from outside repo after editable install.
        candidate = Path.cwd() / args.region_yaml
        if candidate.exists():
            region_yaml = candidate
    if args.command == "smoke" and args.limit is None:
        args.limit = 5

    if args.command in {"run", "smoke"}:
        summary = run_pipeline(
            input_path=args.input,
            region=args.region,
            output_dir=args.output,
            cfg=cfg,
            region_yaml=region_yaml,
            limit=args.limit,
            segmenter_backend=args.segmenter,
            skip_classification=args.skip_classification,
            bioclip_model=args.bioclip_model,
        )
        print("Done. Summary:")
        for key, value in summary.items():
            if key != "species_breakdown":
                print(f"  {key}: {value}")
        print(f"Outputs saved to: {args.output}")
    elif args.command == "eval":
        result = run_eval(
            dataset=args.dataset,
            split=args.split,
            region=args.region,
            output_dir=args.output,
            cfg=cfg,
            region_yaml=region_yaml,
            limit=args.limit,
            segmenter_backend=args.segmenter,
            skip_classification=args.skip_classification,
            bioclip_model=args.bioclip_model,
        )
        print("Done. Metrics:")
        metrics = result["metrics"]
        for key in ["precision", "recall", "f1", "tp", "fp", "fn", "num_images"]:
            print(f"  {key}: {metrics[key]}")
        print(f"Outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
