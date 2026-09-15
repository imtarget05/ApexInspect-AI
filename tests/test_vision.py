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
        # Warmup: first ONNX run includes one-time costs (memory arena,
        # graph init). Benchmarks must measure steady-state latency.
        try:
            warm_frame, warm_gt = self.simulator.generate_pcb_frame(inject_defect=True, specific_defect="short_circuit")
            self.detector.infer(warm_frame, warm_gt)
        except Exception:
            pass

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
        """Detector must run inference and return annotated frame with steady-state latency <= 120ms."""
        frame, ground_truth = self.simulator.generate_pcb_frame(inject_defect=True, specific_defect="short_circuit")
        latencies = []
        detections = []
        annotated_frame = None
        for _ in range(3):
            annotated_frame, detections, latency_ms = self.detector.infer(frame, ground_truth)
            latencies.append(latency_ms)
        steady_ms = sum(latencies) / len(latencies)

        self.assertIsInstance(annotated_frame, np.ndarray)
        self.assertEqual(annotated_frame.shape, (640, 640, 3))
        self.assertGreaterEqual(len(detections), 1)
        self.assertLessEqual(steady_ms, 120.0, f"Steady-state latency {steady_ms:.1f}ms (runs={['%.1f' % v for v in latencies]}) exceeded 120ms requirement!")

    def test_canonical_onnx_model_is_deployed(self):
        """Colab-trained weights must be deployed at models/yolov8n_pcb_defect.onnx."""
        canonical = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n_pcb_defect.onnx"))
        self.assertTrue(os.path.exists(canonical), f"Missing canonical model: {canonical}")
        self.assertGreater(os.path.getsize(canonical), 1_000_000, "Canonical ONNX looks truncated (<1MB)")

    def test_detector_resolves_canonical_model_and_reads_env_threshold(self):
        """Detector must resolve models/ path and honor CONFIDENCE_THRESHOLD env."""
        self.assertTrue(self.detector.model_path.endswith("models/yolov8n_pcb_defect.onnx"))
        if self.detector.use_onnx:
            info = self.detector.get_model_info()
            self.assertEqual(info["input_shape"], [1, 3, 640, 640])
            self.assertEqual(info["output_shape"], [1, 10, 8400])

    def test_detector_normalizes_training_label_short(self):
        """Raw training label 'short' (data.yaml) must map to canonical 'short_circuit'."""
        self.assertEqual(self.detector.normalize_class("short"), "short_circuit")
        self.assertEqual(self.detector.normalize_class("spur"), "spur")

if __name__ == "__main__":
    unittest.main()
