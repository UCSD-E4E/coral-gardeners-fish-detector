from __future__ import annotations


def prompts_for_species(scientific_name: str) -> list[str]:
    return [
        f"a photo of a {scientific_name}",
        f"an underwater photo of a {scientific_name}",
        f"a reef fish species {scientific_name}",
        f"{scientific_name}, a reef fish",
    ]


def build_species_prompt_map(species: list[str]) -> dict[str, list[str]]:
    return {name: prompts_for_species(name) for name in species}
