from coral_fish_pipeline.utils.boxes import iou_xyxy
from coral_fish_pipeline.utils.boxes import non_max_suppression
from coral_fish_pipeline.evaluation.metrics import precision_recall_f1, match_predictions_to_ground_truth
from coral_fish_pipeline.models import Detection


def test_iou():
    assert iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert round(iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15]), 2) == 0.14


def test_nms_removes_duplicate_boxes():
    detections = [
        Detection(image_id="img", det_id="high", bbox_xyxy=[0, 0, 10, 10], score=0.9),
        Detection(image_id="img", det_id="low", bbox_xyxy=[1, 1, 11, 11], score=0.8),
    ]

    kept = non_max_suppression(detections, iou_threshold=0.5)

    assert [d.det_id for d in kept] == ["high"]


def test_nms_keeps_separate_boxes():
    detections = [
        Detection(image_id="img", det_id="a", bbox_xyxy=[0, 0, 10, 10], score=0.9),
        Detection(image_id="img", det_id="b", bbox_xyxy=[30, 30, 40, 40], score=0.8),
    ]

    kept = non_max_suppression(detections, iou_threshold=0.5)

    assert [d.det_id for d in kept] == ["a", "b"]


def test_match_and_metrics():
    matches, fp, fn = match_predictions_to_ground_truth([[0,0,10,10]], [[0,0,10,10]], 0.5)
    assert len(matches) == 1 and not fp and not fn
    m = precision_recall_f1(1, 1, 0)
    assert round(m["precision"], 2) == 0.5
