import pathlib


def test_training_yaml_has_separate_test():
    import yaml
    with open("models/training_data.yaml") as f:
        d = yaml.safe_load(f)
    assert d["test"] != d["val"]
    assert "production_val" in d["test"]


def _train_src() -> str:
    return pathlib.Path("models/train.py").read_text(encoding="utf-8")


def test_train_script_trains_from_canonical_training_yaml():
    """train.py must consume models/training_data.yaml, not a stray CWD data.yaml.

    plan-04 Task 3 claimed train.py was unified with the canonical config, but the
    script still looked for `data.yaml` in the working directory and regenerated
    its own copy - so the tracked SoT (splits + `test: images/production_val`)
    was silently ignored at training time.
    """
    src = _train_src()
    assert "training_data.yaml" in src, "train.py must reference models/training_data.yaml"
    assert 'data_yaml = "data.yaml"' not in src, (
        "train.py must not fall back to a stray CWD data.yaml"
    )


def test_train_script_does_not_hardcode_gpu_device():
    """device=0 breaks CPU-only hosts (review-plan-04 finding #5)."""
    src = _train_src()
    assert "device=0," not in src, "train.py must not hardcode CUDA device 0"
    assert "cuda.is_available()" in src, "train.py should auto-detect the compute device"


def test_train_script_exports_to_canonical_onnx_path():
    """The exported ONNX must land where the Detector resolves it."""
    src = _train_src()
    assert "yolov8n_pcb_defect.onnx" in src
