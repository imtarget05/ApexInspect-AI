import os
import sys
import unittest
import numpy as np

# Ensure root directory is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.simulator import PCBCameraSimulator
from src.vision.detector import PCBDefectDetector

class TestVisionPipeline(unittest.TestCase):
    """Unit tests for industrial vision simulation and ONNX defect detector."""

    def setUp(self):
        self.simulator = PCBCameraSimulator(width=640, height=640)
        self.detector = PCBDefectDetector()

    def test_simulator_generates_valid_pcb_frame(self):
        """Simulator must generate a 640x640x3 image frame."""
        frame, defects = self.simulator.generate_pcb_frame(inject_defect=False)
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (640, 640, 3))
        self.assertEqual(len(defects), 0)

    def test_simulator_injects_defects(self):
        """Simulator must inject specific defect classes with bounding boxes."""
        defect_classes = ["short_circuit", "missing_hole", "mouse_bite", "open_circuit", "spur"]
        for def_type in defect_classes:
            frame, defects = self.simulator.generate_pcb_frame(inject_defect=True, specific_defect=def_type)
            self.assertGreaterEqual(len(defects), 1)
            self.assertEqual(defects[0]["class"], def_type)
            self.assertEqual(len(defects[0]["bbox"]), 4)
            self.assertGreaterEqual(defects[0]["confidence"], 0.70)

    def test_detector_inference_latency_under_threshold(self):
        """Detector must run inference and return annotated frame with latency <= 50ms."""
        frame, ground_truth = self.simulator.generate_pcb_frame(inject_defect=True, specific_defect="short_circuit")
        annotated_frame, detections, latency_ms = self.detector.infer(frame, ground_truth)

        self.assertIsInstance(annotated_frame, np.ndarray)
        self.assertEqual(annotated_frame.shape, (640, 640, 3))
        self.assertGreaterEqual(len(detections), 1)
        self.assertLessEqual(latency_ms, 50.0, f"Inference latency {latency_ms}ms exceeded 50ms requirement!")

if __name__ == "__main__":
    unittest.main()
