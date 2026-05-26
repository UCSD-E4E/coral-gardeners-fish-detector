from coral_fish_pipeline.utils.boxes import expand_box_xyxy, clip_box_xyxy


def test_expand_box_clips():
    box = expand_box_xyxy([5, 5, 10, 10], 0.5, width=12, height=12, min_crop_size=1)
    assert box[0] >= 0 and box[1] >= 0
    assert box[2] <= 12 and box[3] <= 12
    assert box[2] > box[0] and box[3] > box[1]
