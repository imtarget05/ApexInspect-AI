import time
import threading
from typing import Optional, Tuple, Dict, Any, Union
import cv2
import numpy as np

from .quality import check_frame


class IndustrialCameraStream:
    """
    High-throughput Threaded Industrial Camera Ingestion.
    Supports RTSP streams, GigE/USB3 Industrial Cameras (OpenCV backend), and video files.
    Features:
    - Dedicated capture thread with zero-lag latest-frame policy (discards stale frames).
    - In-flight frame quality inspection (blur, underexposure, overexposure).
    - Resilient automatic reconnection for unstable factory network links.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        validate_quality: bool = True,
        reconnect_interval: float = 1.0,
        max_reconnect_attempts: int = 3,
        fps_limit: Optional[float] = None
    ):
        self.source = source
        self.validate_quality = validate_quality
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.fps_limit = fps_limit

        self.cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        self._last_frame: Optional[np.ndarray] = None
        self._last_quality: Dict[str, Any] = {"ok": True, "reason": "none"}
        self._last_grab_time: float = 0.0
        self._fps_actual: float = 0.0
        self._frames_grabbed = 0

    def _open_capture(self) -> bool:
        """Opens or reopens the video capture device."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source)
        else:
            # RTSP or video file
            self.cap = cv2.VideoCapture(str(self.source))

        if self.cap and self.cap.isOpened():
            return True
        return False

    def start(self) -> "IndustrialCameraStream":
        """Starts the background frame grabber thread."""
        if self._running:
            return self

        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()
        # Brief pause to allow initial frame capture
        time.sleep(0.1)
        return self

    def _capture_worker(self) -> None:
        """Dedicated thread continuously reading the latest frame to eliminate buffer delay."""
        attempts = 0
        while self._running:
            if self.cap is None or not self.cap.isOpened():
                if attempts < self.max_reconnect_attempts:
                    attempts += 1
                    ok = self._open_capture()
                    if not ok:
                        time.sleep(self.reconnect_interval)
                        continue
                    else:
                        attempts = 0
                else:
                    # Exceeded reconnection attempts
                    time.sleep(self.reconnect_interval)
                    continue

            ret, frame = self.cap.read()
            now = time.perf_counter()

            if not ret or frame is None:
                # End of stream or link drop
                attempts += 1
                time.sleep(0.05)
                continue

            # Quality validation
            q_info = {"ok": True, "reason": "skipped"}
            if self.validate_quality:
                try:
                    q_info = check_frame(frame)
                except Exception as e:
                    q_info = {"ok": True, "reason": f"error_{e}"}

            with self._lock:
                self._last_frame = frame
                self._last_quality = q_info
                delta = now - self._last_grab_time if self._last_grab_time > 0 else 0.033
                self._fps_actual = 1.0 / delta if delta > 0 else 30.0
                self._last_grab_time = now
                self._frames_grabbed += 1

            if self.fps_limit and self.fps_limit > 0:
                target_period = 1.0 / self.fps_limit
                elapsed = time.perf_counter() - now
                if target_period > elapsed:
                    time.sleep(target_period - elapsed)

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Retrieves the freshest frame captured by the background thread.
        Returns:
            success: bool
            frame: Optional[np.ndarray]
            metadata: Dict containing quality metrics and stream fps.
        """
        with self._lock:
            if self._last_frame is None:
                return False, None, {"reason": "no_frame_yet"}
            frame_copy = self._last_frame.copy()
            meta = {
                "quality": dict(self._last_quality),
                "fps": round(self._fps_actual, 1),
                "frames_grabbed": self._frames_grabbed
            }
            return True, frame_copy, meta

    def stop(self) -> None:
        """Stops the worker thread and releases video capture hardware."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def release(self) -> None:
        """Alias for stop()."""
        self.stop()

    def __enter__(self) -> "IndustrialCameraStream":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
