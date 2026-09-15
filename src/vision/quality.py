import cv2
import numpy as np
from typing import Dict, Any

BLUR_THRESHOLD = 100.0
BRIGHT_MIN = 60.0
BRIGHT_MAX = 200.0
MIN_SIZE = 640

def check_frame(img: np.ndarray) -> Dict[str, Any]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    if w < MIN_SIZE or h < MIN_SIZE:
        return {"ok": False, "reason": "resolution", "laplacian_var": lap_var, "brightness": brightness, "width": w, "height": h}
    if lap_var < BLUR_THRESHOLD:
        return {"ok": False, "reason": "blur", "laplacian_var": lap_var, "brightness": brightness, "width": w, "height": h}
    if brightness < BRIGHT_MIN or brightness > BRIGHT_MAX:
        return {"ok": False, "reason": "brightness", "laplacian_var": lap_var, "brightness": brightness, "width": w, "height": h}
    return {"ok": True, "reason": "ok", "laplacian_var": lap_var, "brightness": brightness, "width": w, "height": h}
