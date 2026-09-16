import os
import sys
import unittest
import tempfile
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.stream import IndustrialCameraStream


class TestIndustrialCameraStream(unittest.TestCase):
    """Verifies threaded camera stream ingestion and quality check integration."""

    def setUp(self):
        # Create a temporary synthetic video file for deterministic offline testing
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.temp_dir, "test_stream.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, 20.0, (640, 640))
        for _ in range(15):
            # Create a green PCB-like frame with texture
            frame = np.full((640, 640, 3), (34, 139, 34), dtype=np.uint8)
            cv2.rectangle(frame, (100, 100), (300, 300), (0, 200, 200), 2)
            cv2.putText(frame, "PCB TRACE TEST", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            out.write(frame)
        out.release()

    def tearDown(self):
        if os.path.exists(self.video_path):
            try:
                os.remove(self.video_path)
            except Exception:
                pass
        if os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except Exception:
                pass

    def test_threaded_camera_stream_reads_frames(self):
        """Threaded grabber must successfully ingest frames from video stream with quality metrics."""
        stream = IndustrialCameraStream(
            source=self.video_path,
            validate_quality=True,
            fps_limit=30.0
        )
        stream.start()

        # Allow worker thread to grab frames
        time.sleep(0.3)

        success, frame, meta = stream.read()
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (640, 640, 3))
        self.assertIn("quality", meta)
        self.assertIn("fps", meta)
        self.assertGreater(meta["frames_grabbed"], 0)

        stream.stop()

    def test_context_manager_lifecycle(self):
        """Stream should cleanly start and stop via Python context manager."""
        with IndustrialCameraStream(source=self.video_path, validate_quality=False) as stream:
            time.sleep(0.2)
            success, frame, meta = stream.read()
            self.assertTrue(success)
            self.assertIsNotNone(frame)

        # After exiting context manager, stream must be stopped
        self.assertFalse(stream._running)


if __name__ == "__main__":
    unittest.main()
