def test_training_yaml_has_separate_test():
    import yaml
    with open("models/training_data.yaml") as f:
        d = yaml.safe_load(f)
    assert d["test"] != d["val"]
    assert "production_val" in d["test"]
