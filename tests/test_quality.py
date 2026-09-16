import os

import cv2
import numpy as np
import pytest
from src.vision.quality import check_frame


def _sharp_frame(size=640, level=127, line_val=200):
    """High-contrast copper traces on a flat board: sharp, easy to score."""
    img = np.full((size, size, 3), level, dtype=np.uint8)
    for i in range(20, size, 40):
        cv2.line(img, (i, 10), (i, size - 10), (line_val, line_val, line_val), 1)
        cv2.line(img, (10, i), (size - 10, i), (line_val, line_val, line_val), 1)
    return img


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


def test_verdict_is_resolution_invariant():
    """A sharp board must not be called blurry just because the sensor is big.

    Real PCB capture is ~2240x2016. Measuring Laplacian variance on the native
    frame scored 75.6 (=> "blur") for content that scores 1313.9 at the 640px
    scale the detector actually feeds the network, which silently rejected
    every production frame. Blur must be measured at the inference scale.
    """
    sharp_640 = _sharp_frame()
    big = cv2.resize(sharp_640, (2240, 2016), interpolation=cv2.INTER_LINEAR)

    small_result = check_frame(sharp_640)
    big_result = check_frame(big)

    assert small_result["ok"] is True
    assert big_result["ok"] is True, (
        f"sharp 2240x2016 frame rejected as {big_result['reason']!r} "
        f"(native lap_var={big_result['native_laplacian_var']:.1f}, "
        f"scaled lap_var={big_result['laplacian_var']:.1f})"
    )


def test_accepts_dark_industrial_camera_frames():
    """Real PCB camera frames are dark (measured mean 55.5-67.7 on 693 images),
    so the old brightness floor of 60 rejected most of the dataset."""
    dark = _sharp_frame(level=52)
    assert dark.mean() < 60, "fixture must reproduce the real dark exposure"

    result = check_frame(dark)

    assert result["ok"] is True, f"dark production frame rejected as {result['reason']!r}"
    assert result["brightness"] < 60


def test_still_rejects_truly_dark_frame():
    """The floor still catches an unlit / obscured lens."""
    result = check_frame(_sharp_frame(level=5))
    assert result["ok"] is False
    assert result["reason"] == "brightness"


def test_still_rejects_low_resolution_frame():
    result = check_frame(_sharp_frame(size=320))
    assert result["ok"] is False
    assert result["reason"] == "resolution"


def test_blur_threshold_sits_inside_measured_evidence_gap():
    """The threshold must stay between real defocus (<=6.4) and the softest
    legit frame (>=88.8), both measured on the 693-image PCB dataset."""
    from src.vision import quality

    assert 6.4 < quality.BLUR_THRESHOLD < 88.8


def test_brightness_bounds_admit_measured_real_camera_exposure():
    """Real PCB frames measure 55.5-67.7 mean brightness."""
    from src.vision import quality

    assert quality.BRIGHT_MIN <= 55.5
    assert quality.BRIGHT_MAX >= 67.7


def test_thresholds_are_env_tunable(monkeypatch):
    """Operators must be able to re-tune the gate per camera without a rebuild."""
    frame = _sharp_frame()
    monkeypatch.setenv("QUALITY_BLUR_THRESHOLD", "999999")
    assert check_frame(frame)["reason"] == "blur"

    monkeypatch.delenv("QUALITY_BLUR_THRESHOLD")
    monkeypatch.setenv("QUALITY_BRIGHT_MIN", "250")
    assert check_frame(frame)["reason"] == "brightness"
