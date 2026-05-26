import json
from pathlib import Path

from PIL import Image, ImageDraw

from coral_fish_pipeline.models import ClassificationResult
from coral_fish_pipeline.pipeline import run_pipeline


class FakeBioCLIPClassifier:
    def __init__(self, *args, **kwargs):
        pass

    def classify_crops(self, crops):
        return [
            ClassificationResult(
                crop_id=crop.crop_id,
                predicted_species="Acanthurus triostegus",
                confidence=0.9,
                top5=[("Acanthurus triostegus", 0.9)],
                region="moorea",
                status="confident",
                model_id="fake-bioclip",
            )
            for crop in crops
        ]


def make_input(tmp_path: Path) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([35, 45, 75, 65], fill=(20, 35, 80))
    image.save(input_dir / "fish.jpg")
    return input_dir


def make_region_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "top25.yaml"
    path.write_text(
        "regions:\n"
        "  moorea:\n"
        "    names:\n"
        "      - Acanthurus triostegus\n",
        encoding="utf-8",
    )
    return path


def make_cfg(**output):
    return {
        "input": {"image_extensions": [".jpg"]},
        "segmentation": {
            "backend": "mock",
            "prompts": ["fish"],
            "min_confidence": 0.25,
            "max_detections_per_image": 10,
        },
        "postprocess": {
            "min_box_area": 20,
            "min_box_side": 2,
            "max_aspect_ratio": 20.0,
            "enable_buoy_filter": False,
            "rope_check_enabled": False,
        },
        "crop": {"padding_percent": 0.2, "min_crop_size": 16, "save_masked_crop": True},
        "classification": {
            "primary_model_id": "hf-hub:imageomics/bioclip-2.5-vith14",
            "fallback_model_id": "hf-hub:imageomics/bioclip-2",
        },
        "preprocessing": {
            "enabled": False,
            "method": "none",
            "apply_to_segmentation": True,
            "apply_to_classification": False,
        },
        "output": {
            "save_crops": False,
            "save_masks": False,
            "save_overlays": False,
            "save_contact_sheet": False,
            **output,
        },
    }


def run_fake_pipeline(tmp_path, monkeypatch, **output):
    monkeypatch.setattr("coral_fish_pipeline.pipeline.BioCLIPClassifier", FakeBioCLIPClassifier)
    input_dir = make_input(tmp_path)
    region_yaml = make_region_yaml(tmp_path)
    output_dir = tmp_path / "output"
    run_pipeline(input_dir, "moorea", output_dir, make_cfg(**output), region_yaml)
    return output_dir


def test_default_run_saves_metadata_only_and_cleans_temp(tmp_path, monkeypatch):
    output_dir = run_fake_pipeline(tmp_path, monkeypatch)

    assert (output_dir / "metadata" / "detections.jsonl").exists()
    assert (output_dir / "metadata" / "classifications.csv").exists()
    assert not (output_dir / "crops").exists()
    assert not (output_dir / "masks").exists()
    assert not (output_dir / "overlays").exists()
    assert not (output_dir / "crops_by_species").exists()
    assert not (output_dir / "contact_sheet.jpg").exists()
    assert not list(output_dir.glob(".tmp_artifacts_*"))

    det = json.loads((output_dir / "metadata" / "detections.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert det["mask_path"] is None


def test_save_crops_persists_crop_outputs(tmp_path, monkeypatch):
    output_dir = run_fake_pipeline(tmp_path, monkeypatch, save_crops=True)

    assert (output_dir / "crops" / "raw").is_dir()
    assert (output_dir / "crops" / "masked").is_dir()
    assert (output_dir / "crops_by_species").is_dir()


def test_save_masks_persists_masks_and_metadata_paths(tmp_path, monkeypatch):
    output_dir = run_fake_pipeline(tmp_path, monkeypatch, save_masks=True)

    assert (output_dir / "masks").is_dir()
    det = json.loads((output_dir / "metadata" / "detections.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert det["mask_path"]
    assert Path(det["mask_path"]).exists()


def test_save_overlays_persists_overlays(tmp_path, monkeypatch):
    output_dir = run_fake_pipeline(tmp_path, monkeypatch, save_overlays=True)

    assert (output_dir / "overlays").is_dir()
    assert list((output_dir / "overlays").glob("*.jpg"))


def test_save_debug_artifacts_config_recreates_full_output(tmp_path, monkeypatch):
    output_dir = run_fake_pipeline(
        tmp_path,
        monkeypatch,
        save_crops=True,
        save_masks=True,
        save_overlays=True,
        save_contact_sheet=True,
    )

    assert (output_dir / "crops").is_dir()
    assert (output_dir / "masks").is_dir()
    assert (output_dir / "overlays").is_dir()
    assert (output_dir / "crops_by_species").is_dir()
    assert (output_dir / "contact_sheet.jpg").exists()
