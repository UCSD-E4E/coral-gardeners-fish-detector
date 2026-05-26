from pathlib import Path
from coral_fish_pipeline.classification.region_loader import load_region_species


def test_load_region_species():
    species = load_region_species(Path("resources/top25.yaml"), "moorea")
    assert len(species) == 25
    assert "Chaetodon auriga" in species
