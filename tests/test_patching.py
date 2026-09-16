import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.detector import PCBDefectDetector
from src.vision.patching import HighResPatchInferencer


class TestHighResPatching(unittest.TestCase):
    """Verifies SAHI-style high-resolution sliding window inference and NMS coordinate merging."""

    def setUp(self):
        self.detector = PCBDefectDetector()
        self.inferencer = HighResPatchInferencer(
            detector=self.detector,
            patch_size=(640, 640),
            overlap_ratio=0.20,
            iou_threshold=0.45
        )

    def test_generate_slices_covers_full_resolution(self):
        """Slice coordinates must completely cover a high-res 1280x1280 canvas."""
        h, w = 1280, 1280
        slices = self.inferencer.generate_slices(h, w)
        self.assertGreater(len(slices), 1)

        # Check all slices are within bounds
        for (x1, y1, x2, y2) in slices:
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, w)
            self.assertLessEqual(y2, h)
            self.assertGreater(x2, x1)
            self.assertGreater(y2, y1)

        # Check corners are covered
        has_origin = any(x1 == 0 and y1 == 0 for (x1, y1, x2, y2) in slices)
        has_bottom_right = any(x2 == w and y2 == h for (x1, y1, x2, y2) in slices)
        self.assertTrue(has_origin)
        self.assertTrue(has_bottom_right)

    def test_generate_slices_small_image_returns_single_tile(self):
        """Image smaller than patch size should produce exactly one full slice."""
        slices = self.inferencer.generate_slices(500, 500)
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0], (0, 0, 500, 500))

    def test_apply_global_nms_deduplicates_overlapping_detections(self):
        """Duplicate detections across patch seams must be merged into one."""
        redundant_dets = [
            {"class": "short_circuit", "confidence": 0.92, "bbox": [100, 100, 150, 150]},
            {"class": "short_circuit", "confidence": 0.88, "bbox": [102, 101, 152, 151]}, # almost identical
            {"class": "mouse_bite", "confidence": 0.80, "bbox": [300, 300, 340, 340]},
        ]
        merged = self.inferencer.apply_global_nms(redundant_dets)
        self.assertEqual(len(merged), 2)
        classes = [m["class"] for m in merged]
        self.assertIn("short_circuit", classes)
        self.assertIn("mouse_bite", classes)

    def test_infer_high_res_end_to_end(self):
        """End-to-end inference on a 1000x1000 simulated frame."""
        # Create a green PCB-like canvas
        high_res_canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)
        high_res_canvas[:, :] = (34, 139, 34)

        ann, dets, lat_ms = self.inferencer.infer_high_res(
            high_res_canvas,
            include_full_image=False
        )
        self.assertEqual(ann.shape, (1000, 1000, 3))
        self.assertIsInstance(dets, list)
        self.assertGreater(lat_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
