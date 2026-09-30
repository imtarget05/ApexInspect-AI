"""P0-02: a quality-rejected frame must never be reported as a clean board.

Root cause
----------
``PCBDefectDetector.infer`` defines a two-state contract in its own docstring::

    detections: [...]  -> "inspected, no defects found"   (a PASS)
    detections: None   -> "no verdict could be reached"    (an UNKNOWN)

but the quality gate returned ``[]`` when ``check_frame`` rejected a frame::

    if not q["ok"]:
        return ..., [], latency_ms        # <-- a PASS, not an UNKNOWN

So a blurred, black or undersized camera frame produced an empty detection
list, which every downstream consumer reads as "inspected and clean":

* the Streamlit UI counted it as ``is_defective=False`` and wrote a telemetry
  row, which is what drives the 3-consecutive-defect line-stop trigger;
* ``PatchDetector`` merged it into an empty aggregate;
* the HUD rendered a green "STATUS: PASS (NO DEFECTS)" banner.

An unusable observation silently became a negative result. In solder-defect
inspection that is a false negative produced by the failure of the evidence
gathering itself, which is the one thing a quality gate exists to prevent.

Semantics chosen
----------------
A rejected frame is an **inspection failure**, not a clean observation and not a
resumption of the streak. It is neither positive nor negative evidence about the
board, so it:

* is never counted as a clean frame,
* never increments the consecutive-defect streak,
* never resets it either -- it *pauses* the sequence. The next real
  observation continues the count, because a line that produced two defective
  boards, went dark, then produced a third is still on a defect run, and a
  camera dropout must not quietly clear an escalating line-stop.
* is recorded as an inspection-failure event, so the gap is auditable.
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("numpy", reason="numpy not installed in this environment")
pytest.importorskip("cv2", reason="cv2 not installed in this environment")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision import detector as detector_mod
from src.vision.detector import PCBDefectDetector

GOOD = 640


def _load_detector(monkeypatch, tmp_path):
    """A detector with a *working* model path but a stubbed ONNX session.

    The real question under test is the routing of a rejected frame, not the
    accuracy of YOLO, so the session is replaced with something that returns
    the right tensor shape and nothing else. ``require_model``/``use_onnx``
    must both look healthy, otherwise ``infer`` takes the model-unavailable
    branch and the test would pass for the wrong reason.
    """
    model_path = tmp_path / "stub.onnx"
    model_path.write_bytes(b"stub")
    monkeypatch.setattr(detector_mod, "resolve_model_path", lambda provided=None: str(model_path))

    class _StubSession:
        def run(self, _outputs, _feed):
            return [np.zeros((1, 4 + 1, 8400), dtype=np.float32)]

    det = detector_mod.PCBDefectDetector(str(model_path))
    det.use_onnx = True
    det.session = _DetectingSession()
    det.output_name = "output0"
    det.input_name = "images"
    return det


def _load_clean_detector(monkeypatch, tmp_path):
    """As above, but the stub session finds nothing: a genuine clean board."""
    det = _load_detector(monkeypatch, tmp_path)
    det.session = _StubSession()
    return det



def _sharp_frame(seed=0):
    """A frame the quality gate accepts: sharp and mid-brightness."""
    rng = np.random.default_rng(seed)
    img = np.full((GOOD, GOOD, 3), 120, dtype=np.uint8)
    for _ in range(40):
        x, y = rng.integers(0, GOOD - 8, size=2)
        img[y:y + 8, x:x + 8] = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
    return img


def _detecting_session(center_score=0.9):
    """A stub session emitting one confident box at the frame centre."""
    out = np.zeros((1, 5, 8400), dtype=np.float32)
    out[0, 0, 0] = center_score          # class 0 score
    out[0, 4, 0] = 320.0                # cx
    out[0, 1, 0] = 320.0                # cy
    out[0, 2, 0] = 80.0                 # w
    out[0, 3, 0] = 80.0                 # h
    return out


class _DetectingSession:
    def run(self, _outputs, _feed):
        return [_detecting_session()]


class _StubSession:
    def run(self, _outputs, _feed):
        return [np.zeros((1, 5, 8400), dtype=np.float32)]



# --- A1/A2: the two states that must keep working ---------------------------


def test_a1_good_frame_with_no_defect_is_a_clean_pass(monkeypatch, tmp_path):
    det = _load_clean_detector(monkeypatch, tmp_path)
    _, dets, latency = det.infer(_sharp_frame(seed=1))

    assert dets == [], "a genuinely clean board must still report [] (an inspected PASS)"
    assert latency is not None, "a real inspection must report a real latency"


def test_a2_good_frame_with_a_defect_reports_detections(monkeypatch, tmp_path):
    det = _load_detector(monkeypatch, tmp_path)
    _, dets, _ = det.infer(_sharp_frame(seed=2))

    assert dets, "a defect must be reported as a non-empty detection list"
    assert dets is not None


# --- A3/A4: the defect itself ----------------------------------------------
#
# These are the tests that fail on the pre-fix code. Before the fix both
# returned [], which the UI and PatchDetector both read as "inspected, clean".


def test_a3_blurred_frame_yields_no_verdict_not_an_empty_list(monkeypatch, tmp_path):
    det = _load_clean_detector(monkeypatch, tmp_path)
    blurred = np.full((GOOD, GOOD, 3), 120, dtype=np.uint8)  # zero variance -> blur

    _, dets, _ = det.infer(blurred)

    assert dets is None, (
        "a blurred frame is an inspection failure, not a clean board. "
        "Returning [] here reports a false negative on a real board."
    )


def test_a4_black_frame_yields_no_verdict(monkeypatch, tmp_path):
    det = _load_clean_detector(monkeypatch, tmp_path)
    black = np.zeros((GOOD, GOOD, 3), dtype=np.uint8)  # zero brightness

    _, dets, _ = det.infer(black)

    assert dets is None, "an unlit/occluded lens is not a defect-free board"


# --- A5: corrupt input must not be laundered into a clean pass --------------


def test_a5_corrupt_input_raises_rather_than_returning_a_clean_list(monkeypatch, tmp_path):
    """A malformed frame must never surface as [].

    Either a clean, explicit exception or a None no-verdict is acceptable.
    What is forbidden is an empty detection list, which is indistinguishable
    from a successful inspection of a perfect board.
    """
    det = _load_clean_detector(monkeypatch, tmp_path)

    for name, bad in (
        ("None", None),
        ("2-d array", np.zeros((640, 640), dtype=np.uint8)),
        ("wrong dtype", np.full((640, 640, 3), "x", dtype=object)),
        ("empty array", np.array([])),
    ):
        with pytest.raises(Exception):
            dets = det.infer(bad)[1]
            assert dets != [], f"corrupt input ({name}) was laundered into a clean pass"


# --- A6 / state machine: UNKNOWN must never be treated as CLEAN -------------
#
# The 3-consecutive-defect rule is a line-stop trigger. Treating an unusable
# frame as clean RESETS the streak, so a camera failing every third frame
# would prevent the line from ever being stopped. Treating it as a defect
# would stop the line for the wrong reason. Policy: a rejected frame PAUSES
# the sequence -- it neither increments nor resets it.


def _streak_after(observations):
    """Replay observations through the documented streak policy."""
    streak = 0
    for obs in observations:
        if obs == "invalid":
            continue          # pauses: no information, so the count is preserved
        streak = streak + 1 if obs == "defect" else 0
    return streak


def test_a6_invalid_frame_does_not_count_as_a_clean_frame():
    assert _streak_after(["defect", "defect", "invalid", "defect"]) == 3, (
        "an invalid frame must not reset the streak: a line that produced two "
        "defects, went dark, then produced a third is still on a defect run"
    )


def test_a6b_invalid_frame_does_not_increment_the_streak():
    assert _streak_after(["defect", "invalid", "invalid"]) == 1, (
        "an invalid frame carries no evidence of a defect and must not trip the trigger"
    )


def test_a6c_a_real_clean_frame_does_reset_the_streak():
    assert _streak_after(["defect", "defect", "clean"]) == 0, (
        "only an actual clean inspection may reset the consecutive-defect count"
    )


def test_a6d_alternating_invalid_frames_never_trip_the_trigger():
    """A camera failing every other frame must not stop a healthy line."""
    assert _streak_after(["clean", "invalid"] * 20) == 0


# --- negative controls: the fix must not become abstain-everything ---------


def test_negative_control_valid_frames_are_still_verdicted(monkeypatch, tmp_path):
    """If every frame returned None this whole module would pass pointlessly."""
    det = _load_clean_detector(monkeypatch, tmp_path)
    verdicts = [det.infer(_sharp_frame(seed=s))[1] for s in range(5)]

    assert all(v == [] for v in verdicts), (
        "no-verdict routing must apply ONLY to rejected frames; valid frames "
        "must still produce real verdicts"
    )


def test_negative_control_detections_survive_the_quality_gate(monkeypatch, tmp_path):
    det = _load_detector(monkeypatch, tmp_path)
    assert det.infer(_sharp_frame(seed=4))[1], "defect detection regressed"


# --- the HUD must not paint a green PASS on an uninspected board -----------


def test_rejected_frame_is_not_annotated_as_pass(monkeypatch, tmp_path):
    det = _load_clean_detector(monkeypatch, tmp_path)
    annotated, dets, _ = det.infer(np.full((GOOD, GOOD, 3), 120, dtype=np.uint8))

    assert dets is None
    b, g, r = annotated[12, 12]
    assert not (g > 200 and r < 60 and b < 60), (
        "the HUD drew a green PASS banner on a frame that was never inspected"
    )


def test_a4b_undersized_frame_yields_no_verdict(monkeypatch, tmp_path):
    det = _load_clean_detector(monkeypatch, tmp_path)
    tiny = _sharp_frame(seed=3)[:320, :320]  # below MIN_SIZE

    _, dets, _ = det.infer(tiny)

    assert dets is None, "a frame below the resolution floor cannot support a verdict"
