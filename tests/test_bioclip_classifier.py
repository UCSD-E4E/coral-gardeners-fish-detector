import pytest
import torch
from PIL import Image

from coral_fish_pipeline.classification.bioclip_classifier import BioCLIPClassifier


def test_constructor_accepts_allow_original_bioclip_config_key():
    classifier = BioCLIPClassifier(
        species=["Acanthurus triostegus"],
        region="moorea",
        allow_original_bioclip=False,
    )

    assert classifier.primary_model_id == "hf-hub:imageomics/bioclip-2.5-vith14"
    assert classifier.fallback_model_id == "hf-hub:imageomics/bioclip-2"


@pytest.mark.parametrize(
    "model_id",
    [
        "hf-hub:imageomics/bioclip",
        "imageomics/bioclip",
        "hf-hub:imageomics/bioclip@main",
    ],
)
def test_original_bioclip_is_rejected(model_id):
    with pytest.raises(ValueError, match="Original BioCLIP is forbidden"):
        BioCLIPClassifier(
            species=["Acanthurus triostegus"],
            region="moorea",
            primary_model_id=model_id,
        )


def test_original_bioclip_requires_explicit_opt_in():
    classifier = BioCLIPClassifier(
        species=["Acanthurus triostegus"],
        region="moorea",
        primary_model_id="hf-hub:imageomics/bioclip",
        fallback_model_id="hf-hub:imageomics/bioclip-2",
        allow_original_bioclip=True,
    )

    assert classifier.primary_model_id == "hf-hub:imageomics/bioclip"


@pytest.mark.parametrize(
    "model_id",
    [
        "hf-hub:imageomics/bioclip-2",
        "hf-hub:imageomics/bioclip-2.5-vith14",
    ],
)
def test_bioclip_2_models_are_allowed(model_id):
    classifier = BioCLIPClassifier(
        species=["Acanthurus triostegus"],
        region="moorea",
        primary_model_id=model_id,
        fallback_model_id="hf-hub:imageomics/bioclip-2",
    )

    assert classifier.primary_model_id == model_id


class _FakeScaledModel:
    logit_scale = torch.tensor(100.0).log()

    def encode_image(self, _tensor):
        return torch.tensor([[1.0, 0.0]])


def test_predict_image_applies_openclip_logit_scale(tmp_path):
    image_path = tmp_path / "crop.jpg"
    Image.new("RGB", (8, 8), "white").save(image_path)

    classifier = BioCLIPClassifier(
        species=["target fish", "other fish"],
        region="moorea",
        primary_model_id="hf-hub:imageomics/bioclip-2",
        fallback_model_id="hf-hub:imageomics/bioclip-2",
        device="cpu",
    )
    classifier.model = _FakeScaledModel()
    classifier.model_id = "hf-hub:imageomics/bioclip-2"
    classifier.device = torch.device("cpu")
    classifier.preprocess = lambda _image: torch.zeros(3, 8, 8)
    classifier.text_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    pred = classifier._predict_image(image_path)

    assert pred["species"] == "target fish"
    assert pred["confidence"] > 0.99
    assert pred["top1_probability"] > 0.99
    assert pred["margin"] > 0.99
