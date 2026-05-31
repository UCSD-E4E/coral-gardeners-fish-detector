from pathlib import Path
import importlib.util


SCRIPT_PATH = Path("scripts/build_top25_region_prior.py")


def load_script_module():
    spec = importlib.util.spec_from_file_location("build_top25_region_prior", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_make_region_key():
    module = load_script_module()
    assert module.make_region_key("Gulf of Thailand") == "gulf_of_thailand"
    assert module.make_region_key("region_-17.54_-149.83") == "region_17_54_149_83"


def test_write_and_load_regions_yaml(tmp_path):
    module = load_script_module()
    path = tmp_path / "top25.yaml"
    module.write_regions_yaml(path, {"moorea": ["Chaetodon auriga", "Acanthurus triostegus"]})

    assert module.load_existing_regions(path) == {
        "moorea": ["Chaetodon auriga", "Acanthurus triostegus"],
    }
    assert "nc: 2" in path.read_text(encoding="utf-8")
