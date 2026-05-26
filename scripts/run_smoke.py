from coral_fish_pipeline.cli import main

if __name__ == "__main__":
    main([
        "smoke",
        "--input", "data/examples",
        "--region", "moorea",
        "--output", "outputs/smoke_mock",
        "--segmenter", "mock",
        "--skip-classification",
    ])
