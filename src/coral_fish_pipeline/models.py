from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal


@dataclass
class Detection:
    image_id: str
    det_id: str
    bbox_xyxy: list[float]
    mask_path: str | None = None
    score: float = 1.0
    prompt: str | None = None
    status: Literal["accepted", "rejected"] = "accepted"
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CropRecord:
    crop_id: str
    image_id: str
    det_id: str
    raw_crop_path: str
    masked_crop_path: str | None
    bbox_original: list[float]
    bbox_padded: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationResult:
    crop_id: str
    predicted_species: str
    confidence: float
    top5: list[tuple[str, float]]
    region: str
    status: Literal["confident", "uncertain", "unknown", "non_fish"]
    model_id: str
    raw_prediction: dict[str, Any] | None = None
    masked_prediction: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
