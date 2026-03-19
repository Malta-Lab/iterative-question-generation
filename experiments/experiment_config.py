class ExperimentConfig:

    DATASETS = [
        {
            "name": "averitec_dev",
            "path": "datasets/averitec/dev.json"
        },
        {
            "name": "averitec_train",
            "path": "datasets/averitec/train.json"
        },
        {
            "name": "averitec_test",
            "path": "datasets/averitec/test.json"
        }
    ]

    MAX_SAMPLES = None  # ou 100