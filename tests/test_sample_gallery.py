"""Authenticity guards for the curated gallery in `data/sample_pcbs/`.

The first version of this gallery (commit cf22bf3) claimed to hold "real
industrial PCB dataset images" but was in fact six synthetic code-renders of a
single template (identical Laplacian variance 1577.6-1578.3, identical 140 KB
file sizes, dHash Hamming distance >= 42/256 from every real board). These tests
pin the provenance and variability of the gallery so that regression cannot be
shipped as "real data" again.
"""
import json
import os

import cv2
import numpy as np

GALLERY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_pcbs"))
META_PATH = os.path.join(GALLERY, "sample_meta.json")
CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short_circuit", "spur", "spurious_copper"]


def _meta():
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_gallery_covers_every_defect_class_and_pass():
    meta = _meta()
    for cls in CLASSES + ["pass"]:
        assert any(v["class"] == cls for v in meta.values()), f"gallery missing class {cls}"


def test_every_sample_records_real_dataset_provenance():
    """Each frame must trace back to a high-resolution Kaggle PCB capture."""
    for name, m in _meta().items():
        assert m.get("source", "").startswith("images/"), f"{name}: missing dataset source"
        native_w, native_h = m["native_size"]
        assert min(native_w, native_h) >= 1500, (
            f"{name}: native {native_w}x{native_h} is not a real capture - "
            "the previous fake gallery shipped 640px renders"
        )
        img = cv2.imread(os.path.join(GALLERY, name))
        assert img is not None and img.shape[:2] == (640, 640)


def test_ground_truth_boxes_are_inside_the_frame():
    for name, m in _meta().items():
        if m["class"] == "pass":
            assert m["defects"] == [], f"{name}: PASS sample must carry no ground truth"
            continue
        assert m["defects"], f"{name}: defect sample must carry ground truth"
        for d in m["defects"]:
            x1, y1, x2, y2 = d["bbox"]
            assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 640, f"{name}: bad bbox {d['bbox']}"
            assert d["class"] in CLASSES, f"{name}: unknown class {d['class']}"


def test_frames_are_not_copies_of_one_template():
    """Real captures must differ from each other (the fake gallery did not)."""
    meta = _meta()
    names = [n for n, m in meta.items() if m["class"] != "pass"][:6]
    assert len(names) >= 4, "gallery too small to compare variability"
    frames = [cv2.imread(os.path.join(GALLERY, n), cv2.IMREAD_GRAYSCALE).astype(np.float32) for n in names]
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            diff = float(np.abs(frames[i] - frames[j]).mean())
            assert diff > 5.0, f"{names[i]} and {names[j]} look like the same render (mean diff {diff:.2f})"


def test_simulator_streams_real_gallery_frames():
    from src.vision.simulator import PCBCameraSimulator

    sim = PCBCameraSimulator(width=640, height=640)
    frame, gt = sim.generate_pcb_frame(inject_defect=True, specific_defect="short_circuit")
    assert frame.shape == (640, 640, 3)
    assert len(gt) >= 1 and all(d["class"] == "short_circuit" for d in gt)

    pass_frame, pass_gt = sim.generate_pcb_frame(inject_defect=False)
    assert pass_frame.shape == (640, 640, 3)
    assert pass_gt == []