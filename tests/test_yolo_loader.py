from coral_fish_pipeline.utils.boxes import yolo_to_xyxy


def test_yolo_to_xyxy_center():
    box = yolo_to_xyxy(0.5, 0.5, 0.2, 0.4, 100, 200)
    assert box == [40.0, 60.0, 60.0, 140.0]
