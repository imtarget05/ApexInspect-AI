import cv2
import numpy as np
from src.vision.quality import check_frame

def test_reject_blurred_frame():
    sharp = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    cv2.rectangle(sharp, (100, 100), (500, 500), (255, 255, 255), -1)
    blurred = cv2.GaussianBlur(sharp, (51, 51), 0)
    result = check_frame(blurred)
    assert result["ok"] is False
    assert result["reason"] == "blur"

def test_accept_good_frame():
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (500, 500), (255, 255, 255), -1)
    result = check_frame(img)
    assert result["ok"] is True
