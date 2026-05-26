from pathlib import Path
import importlib.util

from PIL import Image


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gbif = load_script("download_gbif_region_media")
folder_eval = load_script("evaluate_bioclip_folder_dataset")


def test_slug_and_unslug_species():
    assert gbif.slug_species("Acanthurus triostegus") == "Acanthurus_triostegus"
    assert gbif.unslug_species("Acanthurus_triostegus") == "Acanthurus triostegus"


def test_license_normalization_and_filtering():
    allowed = {"CC0_1_0", "CC_BY_4_0", "CC_BY_NC_4_0"}

    assert gbif.normalize_license("https://creativecommons.org/licenses/by/4.0/") == "CC_BY_4_0"
    assert gbif.normalize_license("CC-BY-NC-4.0") == "CC_BY_NC_4_0"
    assert gbif.is_allowed_license("CC-BY-4.0", allowed)
    assert not gbif.is_allowed_license("CC-BY-SA-4.0", allowed)


def test_infer_extension_prefers_content_type_then_url():
    assert gbif.infer_extension("image/jpeg", "https://example.org/media") == ".jpg"
    assert gbif.infer_extension(None, "https://example.org/media.png") == ".png"
    assert gbif.infer_extension("text/html", "https://example.org/media") is None


def test_folder_dataset_image_discovery(tmp_path: Path):
    species_dir = tmp_path / "Acanthurus_triostegus"
    other_dir = tmp_path / "Not_in_region"
    species_dir.mkdir()
    other_dir.mkdir()
    Image.new("RGB", (4, 4), "white").save(species_dir / "one.jpg")
    Image.new("RGB", (4, 4), "white").save(other_dir / "two.jpg")

    rows = folder_eval.discover_folder_images(tmp_path, ["Acanthurus triostegus"])

    assert rows == [(species_dir / "one.jpg", "Acanthurus triostegus")]


def test_folder_dataset_metric_calculation():
    rows = [
        {
            "true_species": "Acanthurus triostegus",
            "confidence": 0.9,
            "top1_correct": True,
            "top5_correct": True,
            "skip_reason": "",
        },
        {
            "true_species": "Acanthurus triostegus",
            "confidence": 0.4,
            "top1_correct": False,
            "top5_correct": True,
            "skip_reason": "",
        },
    ]

    metrics, per_species = folder_eval.calculate_metrics(rows, candidate_species_count=25)

    assert metrics["images_evaluated"] == 2
    assert metrics["top1_accuracy"] == 0.5
    assert metrics["top5_accuracy"] == 1.0
    assert per_species[0]["species"] == "Acanthurus triostegus"
    assert per_species[0]["top1_accuracy"] == 0.5
