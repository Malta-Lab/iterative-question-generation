def load_models(path="models.txt"):
    with open(path, "r") as f:
        models = [line.strip() for line in f if line.strip()]
    return models