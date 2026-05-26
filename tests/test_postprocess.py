from PIL import Image
from coral_fish_pipeline.models import Detection
from coral_fish_pipeline.segmentation.postprocess import postprocess_detections


def test_reject_too_small():
    img = Image.new("RGB", (100, 100), "white")
    det = Detection(image_id="x", det_id="x_1", bbox_xyxy=[1, 1, 2, 2])
    out = postprocess_detections([det], img, {"min_box_area": 100})
    assert out[0].status == "rejected"
    assert out[0].rejection_reason == "too_small_area"
