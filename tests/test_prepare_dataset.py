"""Tests for scripts/prepare_pcb_dataset.py (Kaggle -> YOLO dataset builder).

Context: the only dataset-construction code used to live in
`notebooks/train_pcb_defect_yolo.ipynb` (cell 4). It created just
`images/train` + `images/val` with Colab-absolute paths, while the tracked SoT
`models/training_data.yaml` declares `test: images/production_val` - a split no
code in the repo could produce. It also split by *image*, but every physical
board is filed under all six class folders (measured: 693 distinct captures of
10 boards), so the same board landed in train *and* val and inflated validation.
"""
import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_pcb_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_pcb_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def prep():
    return _load_module()


def _write_board(root: Path, cls_dir: str, board: str, boxes, size=(800, 600)):
    """Create one image + Pascal VOC xml pair under <root>/images/<cls_dir>."""
    img_dir = root / "images" / cls_dir
    ann_dir = root / "Annotations" / cls_dir
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    w, h = size
    name = f"{board}_{cls_dir.lower()}_01"
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(img_dir / f"{name}.jpg"), img)

    xml = ET.Element("annotation")
    sz = ET.SubElement(xml, "size")
    ET.SubElement(sz, "width").text = str(w)
    ET.SubElement(sz, "height").text = str(h)
    for cls, (x1, y1, x2, y2) in boxes:
        obj = ET.SubElement(xml, "object")
        ET.SubElement(obj, "name").text = cls
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(x1)
        ET.SubElement(bb, "ymin").text = str(y1)
        ET.SubElement(bb, "xmax").text = str(x2)
        ET.SubElement(bb, "ymax").text = str(y2)
    ET.ElementTree(xml).write(ann_dir / f"{name}.xml")
    return name


def test_voc_to_yolo_lines_normalizes_to_unit_range(prep, tmp_path):
    name = _write_board(tmp_path, "Missing_hole", "01",
                        [("missing_hole", (100, 50, 300, 250))], size=(400, 200))
    lines = prep.voc_to_yolo_lines(tmp_path / "Annotations" / "Missing_hole" / f"{name}.xml",
                                   400, 200)
    assert len(lines) == 1
    cls_id, x, y, w, h = lines[0].split()
    assert cls_id == "0"
    assert float(x) == pytest.approx(0.5)
    assert float(y) == pytest.approx(0.75)
    assert float(w) == pytest.approx(0.5)
    assert float(h) == pytest.approx(1.0)


def test_voc_to_yolo_lines_maps_short_to_class_3(prep, tmp_path):
    name = _write_board(tmp_path, "Short", "02", [("short", (10, 10, 20, 20))])
    lines = prep.voc_to_yolo_lines(tmp_path / "Annotations" / "Short" / f"{name}.xml", 800, 600)
    assert lines and lines[0].startswith("3 ")


def test_voc_to_yolo_lines_skips_unknown_and_clamps(prep, tmp_path):
    name = _write_board(tmp_path, "Spur", "03", [
        ("spur", (10, 10, 30, 30)),
        ("totally_unknown", (0, 0, 10, 10)),
        ("spur", (-50, -50, 40, 40)),
    ])
    lines = prep.voc_to_yolo_lines(tmp_path / "Annotations" / "Spur" / f"{name}.xml", 800, 600)
    assert len(lines) == 2, "unknown class must be dropped"
    for line in lines:
        _, x, y, w, h = line.split()
        for v in (x, y, w, h):
            assert 0.0 <= float(v) <= 1.0


def test_board_splits_are_disjoint_and_deterministic(prep):
    boards = [f"{i:02d}" for i in range(1, 11)]
    first = prep.split_boards(boards, val_ratio=0.2, prod_ratio=0.2, seed=42)
    second = prep.split_boards(boards, val_ratio=0.2, prod_ratio=0.2, seed=42)
    assert first == second, "same seed must give the same split"

    train, val, prod = set(first["train"]), set(first["val"]), set(first["production_val"])
    assert train and val and prod, f"all three splits need boards, got {first}"
    assert not (train & val) and not (train & prod) and not (val & prod), "boards must not leak"
    assert train | val | prod == set(boards), "every board must be assigned exactly once"


def test_build_dataset_matches_training_yaml_declared_splits(prep, tmp_path):
    src = tmp_path / "src"
    for board in ["01", "02", "03", "04", "05"]:
        cls = ["Missing_hole", "Short", "Spur", "Mouse_bite", "Open_circuit"][int(board) - 1]
        _write_board(src, cls, board, [(cls.lower().replace("short", "short"),
                                        (40, 40, 120, 120))])

    out = tmp_path / "dataset_pcb"
    summary = prep.build_yolo_dataset(src, out, val_ratio=0.2, prod_ratio=0.2, seed=42)

    for split in ("train", "val", "production_val"):
        assert (out / "images" / split).is_dir(), f"missing images/{split}"
        assert (out / "labels" / split).is_dir(), f"missing labels/{split}"
    assert (out / "data.yaml").exists()

    import yaml
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["test"] == "images/production_val"
    assert list(cfg["names"].values()) == [
        "missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper",
    ]

    # No physical board may appear in more than one split (group split, not image split).
    seen = {}
    for split, boards in summary["boards_by_split"].items():
        for board in boards:
            assert board not in seen, f"board {board} leaked into {split} and {seen[board]}"
            seen[board] = split

    labelled = sum(len(list((out / "images" / s).glob("*"))) for s in ("train", "val", "production_val"))
    assert labelled == 5, f"all 5 boards must be materialised, got {labelled}"
