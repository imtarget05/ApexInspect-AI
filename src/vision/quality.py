import os
import cv2
import numpy as np
from typing import Dict, Any

# Blur is measured on a 640px-normalised copy of the frame, because Laplacian
# variance is resolution dependent: the same sharp board scores 1313.9 at 640px
# but only 75.6 on a 2240x2016 capture, so a native-resolution threshold
# silently rejected every real production frame. 640 matches NORMALIZE_SIZE in
# the detector pre-processing (what the network actually sees).
# Measured on the 693-image Kaggle PCB dataset at the 640px scale: legit frames
# score >= 88.8 (the 10-per-class boards captured at 1921x2904 are the soft end
# of the set) while deliberately defocused frames score 1.2-6.4. 30 sits inside
# that 14x natural gap: ~4.7x margin above real defocus, ~3x below the softest
# real board.
BLUR_THRESHOLD = 30.0
# Real PCB camera frames are dark: measured brightness over the 693-image Kaggle
# PCB dataset is 55.5-67.7 (mean 58.4), so the previous floor of 60 rejected the
# whole production feed. 40 still catches an unlit or obscured lens.
BRIGHT_MIN = 40.0
BRIGHT_MAX = 200.0
MIN_SIZE = 640
NORMALIZE_SIZE = 640


def _env_float(name: str, default: float) -> float:
    """Read a float override from the environment (per-camera tuning, no rebuild)."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def check_frame(img: np.ndarray) -> Dict[str, Any]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    native_lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    blur_threshold = _env_float("QUALITY_BLUR_THRESHOLD", BLUR_THRESHOLD)
    bright_min = _env_float("QUALITY_BRIGHT_MIN", BRIGHT_MIN)
    bright_max = _env_float("QUALITY_BRIGHT_MAX", BRIGHT_MAX)

    if w < MIN_SIZE or h < MIN_SIZE:
        return {"ok": False, "reason": "resolution", "laplacian_var": native_lap_var,
                "native_laplacian_var": native_lap_var, "brightness": float(gray.mean()),
                "width": w, "height": h}

    if w == NORMALIZE_SIZE and h == NORMALIZE_SIZE:
        norm = gray
    else:
        norm = cv2.resize(gray, (NORMALIZE_SIZE, NORMALIZE_SIZE), interpolation=cv2.INTER_AREA)

    lap_var = float(cv2.Laplacian(norm, cv2.CV_64F).var())
    brightness = float(norm.mean())
    metrics = {"laplacian_var": lap_var, "native_laplacian_var": native_lap_var,
               "brightness": brightness, "width": w, "height": h}

    if lap_var < blur_threshold:
        return {"ok": False, "reason": "blur", **metrics}
    if brightness < bright_min or brightness > bright_max:
        return {"ok": False, "reason": "brightness", **metrics}
    return {"ok": True, "reason": "ok", **metrics}
