import os
import sys
import datetime
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import ingest_pcb_dataset as ingest
from src.backend.database import Base
from src.backend.models import InspectionLog
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def make_tree(root: Path, layout: str):
    if layout == "kaggle":
        for cls in ["Missing_hole", "Spur"]:
            d = root / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
                cv2.imwrite(str(d / f"img{i}.jpg"), img)
    else:
        d = root / "images" / "val"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            cv2.imwrite(str(d / f"v{i}.jpg"), img)


class FakeDetector:
    def infer(self, frame):
        return frame, [{"class": "short", "bbox": [1, 2, 3, 4], "confidence": 0.9}], 12.5


class TestIngest(unittest.TestCase):
    def test_collect_kaggle_layout(self):
        with tempfile.TemporaryDirectory() as td:
            make_tree(Path(td), "kaggle")
            out = ingest.collect_kaggle_images(Path(td))
            self.assertEqual(len(out), 6)

    def test_collect_yolo_layout(self):
        with tempfile.TemporaryDirectory() as td:
            make_tree(Path(td), "yolo")
            out = ingest.collect_yolo_images(Path(td))
            self.assertEqual(len(out), 4)

    def test_collect_prefers_kaggle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_tree(root, "kaggle")
            make_tree(root, "yolo")
            self.assertEqual(len(ingest.collect_dataset_images(root)), 6)

    def test_build_record_normalizes_short(self):
        ts = datetime.datetime(2026, 9, 16, tzinfo=datetime.timezone.utc)
        row = ingest.build_record(
            {"class": "short", "bbox": [1, 2, 3, 4], "confidence": 0.9},
            "pcb-kaggle/Missing_hole/img0.jpg", "SMT-LINE-01", ts, 0, 12.5)
        self.assertTrue(row["is_defective"])
        self.assertEqual(row["defect_classes"], ["short_circuit"])
        self.assertEqual(row["image_filename"], "pcb-kaggle/Missing_hole/img0.jpg")

    def test_run_ingest_dedupes_on_rerun(self):
        eng = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=eng)
        fac = sessionmaker(bind=eng)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_tree(root, "yolo")
            kw = dict(limit=3, line_id="SMT-LINE-01", spacing_s=20,
                      session_factory=fac, detector=FakeDetector())
            r1 = ingest.run_ingest(root, **kw)
            r2 = ingest.run_ingest(root, **kw)
            self.assertEqual(r1["inserted"], 3)
            self.assertEqual((r2["inserted"], r2["skipped"]), (0, 3))
            s = fac()
            try:
                self.assertEqual(s.query(InspectionLog).count(), 3)
            finally:
                s.close()


if __name__ == "__main__":
    unittest.main()
