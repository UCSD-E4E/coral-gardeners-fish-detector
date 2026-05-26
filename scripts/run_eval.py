from coral_fish_pipeline.cli import main

if __name__ == "__main__":
    main([
        "eval",
        "--dataset", "data/yolo_dataset",
        "--split", "test",
        "--region", "moorea",
        "--output", "outputs/eval_mock",
        "--segmenter", "mock",
        "--skip-classification",
    ])
