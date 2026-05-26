import sys
import types

import numpy as np
from PIL import Image

from coral_fish_pipeline.segmentation.sam3_runner import iter_tiles, offset_box_xyxy


def test_tile_coordinate_conversion():
    assert offset_box_xyxy([10, 20, 30, 40], 100, 50, 300, 200) == [110.0, 70.0, 130.0, 90.0]


def test_iter_tiles_covers_image_edges():
    tiles = iter_tiles(width=1000, height=800, tile_size=400, tile_overlap=0.25)

    assert tiles[0] == (0, 0, 400, 400)
    assert any(tile[2] == 1000 for tile in tiles)
    assert any(tile[3] == 800 for tile in tiles)


class _FakeModel:
    def to(self, _device):
        return self

    def eval(self):
        return self


class _FakeProcessor:
    def __init__(self, _model):
        pass

    def set_image(self, image):
        return {"size": image.size}

    def set_text_prompt(self, state, prompt):
        mask = np.zeros((16, 16), dtype=np.float32)
        if prompt == "fish":
            mask[2:8, 3:10] = 1.0
            return {
                "pred_masks": mask[None, None, :, :],
                "pred_boxes": np.array([[3, 2, 10, 8]], dtype=np.float32),
                "logits": np.array([2.0], dtype=np.float32),
            }
        mask[2:8, 3:10] = 1.0
        low_conf_mask = np.zeros((16, 16), dtype=np.float32)
        low_conf_mask[10:13, 10:13] = 1.0
        return {
            "masks": np.stack([mask, low_conf_mask], axis=0),
            "scores": np.array([0.95, 0.1], dtype=np.float32),
        }


def test_sam3_runner_parses_outputs_and_dedupes(monkeypatch, tmp_path):
    builder_mod = types.ModuleType("sam3.model_builder")
    builder_mod.build_sam3_image_model = lambda: _FakeModel()

    processor_mod = types.ModuleType("sam3.model.sam3_image_processor")
    processor_mod.Sam3Processor = _FakeProcessor

    monkeypatch.setitem(sys.modules, "sam3", types.ModuleType("sam3"))
    monkeypatch.setitem(sys.modules, "sam3.model", types.ModuleType("sam3.model"))
    monkeypatch.setitem(sys.modules, "sam3.model_builder", builder_mod)
    monkeypatch.setitem(sys.modules, "sam3.model.sam3_image_processor", processor_mod)

    from coral_fish_pipeline.segmentation.sam3_runner import SAM3Runner

    runner = SAM3Runner(min_confidence=0.25, max_detections_per_image=10)
    detections = runner.predict(
        Image.new("RGB", (16, 16), "white"),
        image_id="img",
        output_mask_dir=tmp_path,
        prompts=["fish", "reef fish"],
    )

    assert len(detections) == 1
    det = detections[0]
    assert det.image_id == "img"
    assert det.bbox_xyxy == [3.0, 2.0, 10.0, 8.0]
    assert det.score > 0.8
    assert det.prompt == "reef fish"
    assert det.mask_path is not None
    assert (tmp_path / "img_0000.png").exists()
