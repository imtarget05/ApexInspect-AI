import pytest

pytest.importorskip("numpy", reason="numpy not installed in this environment")
pytest.importorskip("cv2", reason="cv2 not installed in this environment")

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.detector import ModelUnavailableError, PCBDefectDetector

FRAME = np.zeros((640, 640, 3), dtype=np.uint8)


def _detector_without_model(monkeypatch, model_path):
    """Construct a detector that resolves to a path with no usable model.

    `resolve_model_path` falls back to the canonical committed model whenever
    the requested path is absent, so the resolver itself is patched; otherwise
    these tests would silently load the real model instead.
    """
    import src.vision.detector as detector_mod

    monkeypatch.setattr(detector_mod, "resolve_model_path", lambda provided=None: model_path)
    det = detector_mod.PCBDefectDetector(model_path)
    assert det.use_onnx is False, "test setup: the model should not have loaded"
    return det


# --- missing model ---------------------------------------------------------


def test_missing_model_does_not_return_a_clean_board(tmp_path, monkeypatch):
    det = _detector_without_model(monkeypatch, str(tmp_path / "absent.onnx"))
    annotated, dets, latency = det.infer(FRAME)

    assert dets is None, "a detector with no model must return no verdict"
    assert not isinstance(dets, list), "must not be [] -- [] reads as a clean board"
    assert latency is None, "no inference ran, so no latency may be reported"
    assert det.degraded is True
    assert det.load_error and "not found" in det.load_error
    assert annotated.shape == FRAME.shape


def test_missing_model_renders_unknown_not_pass(tmp_path, monkeypatch):
    """The UNKNOWN banner must be visually distinct from the PASS banner, so a
    degraded frame cannot be mistaken for a clean one."""
    det = _detector_without_model(monkeypatch, str(tmp_path / "absent.onnx"))

    unknown = det.render_annotations(FRAME.copy(), [], None, status="UNKNOWN",
                                     detail="model file not found")
    passed = det.render_annotations(FRAME.copy(), [], 25.0)

    # UNKNOWN banner is amber (0,165,255) BGR; PASS banner is green (0,255,0).
    amber = np.array([0, 165, 255], dtype=np.int16)
    green = np.array([0, 255, 0], dtype=np.int16)
    assert np.abs(unknown[10, 160].astype(np.int16) - amber).sum() <= 20
    assert np.abs(passed[10, 160].astype(np.int16) - green).sum() <= 20
    assert not np.array_equal(unknown, passed)


def test_render_tolerates_a_none_latency(monkeypatch):
    """latency_ms=None must not be formatted as a number (or crash)."""
    det = PCBDefectDetector.__new__(PCBDefectDetector)
    det.use_onnx = False
    det.load_error = "boom"
    out = det.render_annotations(FRAME.copy(), [], None, status="UNKNOWN", detail="boom")
    assert out.shape == FRAME.shape


def test_strict_mode_raises_instead_of_returning_no_verdict(tmp_path, monkeypatch):
    det = _detector_without_model(monkeypatch, str(tmp_path / "absent.onnx"))
    with pytest.raises(ModelUnavailableError):
        det.infer(FRAME, strict=True)


def test_require_model_raises_when_degraded(tmp_path, monkeypatch):
    det = _detector_without_model(monkeypatch, str(tmp_path / "absent.onnx"))
    with pytest.raises(ModelUnavailableError):
        det.require_model()


# --- corrupt model ---------------------------------------------------------


def test_corrupt_model_does_not_report_a_pass(tmp_path, monkeypatch):
    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(b"this is definitely not a valid ONNX graph" * 40)
    det = _detector_without_model(monkeypatch, str(corrupt))

    assert det.load_error and "ONNX load failed" in det.load_error
    _, dets, latency = det.infer(FRAME)
    assert dets is None
    assert latency is None
    with pytest.raises(ModelUnavailableError):
        det.infer(FRAME, strict=True)


# --- the loaded path must be unchanged -------------------------------------


def test_loaded_model_still_returns_a_list_and_a_real_latency(monkeypatch):
    """Guard against over-correction: a working model returns [] plus a number.

    The frame must be one the quality gate ACCEPTS. This used to pass a
    uniform black canvas, which the gate rejects for blur (Laplacian variance
    0) -- so after P0-02 it returned None and this test failed. The assertion
    is still the right one; only the fixture was wrong, because an all-zero
    array is not a photograph of a board. A sharp, mid-brightness textured
    frame is.
    """
    det = PCBDefectDetector()
    if not det.use_onnx:
        pytest.skip("committed model unavailable in this environment")

    rng = np.random.default_rng(11)
    good = np.full((640, 640, 3), 120, dtype=np.uint8)
    for _ in range(40):
        x, y = rng.integers(0, 632, size=2)
        good[y:y + 8, x:x + 8] = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)

    _, dets, latency = det.infer(good)
    assert isinstance(dets, list)
    assert latency is not None and latency > 0.0
    det.require_model()


def test_get_model_info_reports_degraded_state(tmp_path, monkeypatch):
    det = _detector_without_model(monkeypatch, str(tmp_path / "absent.onnx"))
    info = det.get_model_info()
    assert info["use_onnx"] is False
    assert info["degraded"] is True
    assert info["load_error"]


def test_source_no_longer_sleeps_to_fake_a_latency():
    """The 25 ms sleep existed only to make the fabricated HUD look real."""
    src = open(
        os.path.join(os.path.dirname(__file__), "..", "src", "vision", "detector.py"),
        encoding="utf-8",
    ).read()
    assert "time.sleep(0.025)" not in src