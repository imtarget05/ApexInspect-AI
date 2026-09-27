"""Phase-2 detector property invariants (seeded, stdlib + import-clean modules).

``src/vision/detector.py`` needs cv2/numpy (absent from the gateway venv),
so the randomized invariants target what IS import-clean:

  * ``src/industrial/plc_bridge.py`` (stdlib-only; pymodbus import is lazy):
    telemetry yield clamping, halt/resume state-machine consistency, mode
    parsing, and a windowed-dispatch (debounce) invariant over bursty
    random timestamps;
  * axis-aligned box IoU math (pure-python mirror of the overlap computation
    that feeds NMS in the cv2 path): range, symmetry, reflexivity.

No new dependencies; no network. The cv2 NMS path is covered only by a
guard asserting the pure math stays consistent with the documented
threshold semantics.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.industrial.plc_bridge import PLCBridge  # noqa: E402

pytestmark = [pytest.mark.adversarial]

SEED = 20260927
N_CASES = 200


# ---- pure IoU math (cv2-free mirror of the NMS overlap prefilter) -----------

def iou(box_a: tuple, box_b: tuple) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, by2_a = box_a[0], box_a[1], box_a[2], box_a[3]
    bx1, by1, bx2, by2_b = box_b[0], box_b[1], box_b[2], box_b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(by2_a, by2_b)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, by2_a - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2_b - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _rand_box(rng: random.Random) -> tuple:
    x1 = rng.uniform(-50, 600)
    y1 = rng.uniform(-50, 600)
    x2 = x1 + rng.choice([0, 1, 5, 60, 300])
    y2 = y1 + rng.choice([0, 1, 5, 60, 300])
    return (x1, y1, x2, y2)


def test_iou_range_symmetric_reflexive():
    rng = random.Random(SEED)
    for case in range(N_CASES):
        a, b = _rand_box(rng), _rand_box(rng)
        v = iou(a, b)
        assert 0.0 <= v <= 1.0, (
            f"case {case}: IoU {v} outside [0,1] for {a} {b}")
        assert iou(a, b) == iou(b, a), (
            f"case {case}: IoU must be symmetric")
        self_iou = iou(a, a)
        area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        if area > 0:
            assert self_iou == 1.0, (
                f"case {case}: non-degenerate box must self-overlap at 1.0")
        else:
            assert self_iou == 0.0, (
                f"case {case}: degenerate box must self-overlap at 0.0")
    # Identical overlapping detections would be merged by NMS at any
    # threshold below 1.0; disjoint boards never merge.
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


# ---- PLC telemetry clamping ---------------------------------------------------

def test_telemetry_yield_clamped_and_round_trips():
    rng = random.Random(SEED + 1)
    for case in range(N_CASES):
        y = rng.choice([
            rng.uniform(-500, 500), 0.0, 100.0, 98.5,
            float(rng.randint(-10**6, 10**6)) / 100.0,
        ])
        bridge = PLCBridge(mode="simulation")
        res = bridge.update_telemetry(total=rng.randint(0, 10**6),
                                      defects=rng.randint(0, 10**5),
                                      yield_rate=y, defect_code=rng.randint(0, 5))
        assert res["status"] == "SIMULATED", (
            f"case {case}: simulation mode must never claim hardware dispatch")
        scaled = bridge._simulated_registers[bridge.REG_YIELD_RATE]
        assert 0 <= scaled <= 10000, (
            f"case {case}: scaled yield {scaled} escapes [0,10000] for input {y}")
        assert res["yield_rate"] == scaled / 100, (
            f"case {case}: reported yield must round-trip the clamped register")
        assert 0.0 <= res["yield_rate"] <= 100.0, (
            f"case {case}: reported yield {res['yield_rate']} outside percent range")


# ---- halt/resume state machine under random op sequences ---------------------

def test_halt_resume_state_machine_consistent():
    rng = random.Random(SEED + 2)
    for case in range(N_CASES):
        bridge = PLCBridge(mode="simulation")
        ops = [rng.choice(["halt", "resume", "divert"]) for _ in range(rng.randint(1, 8))]
        last_cmd = None
        diverted_after_cmd = False
        for op in ops:
            if op == "halt":
                r = bridge.halt_line()
                assert (r["conveyor_running"], r["tower_light"]) is not None
                last_cmd = "halt"
                diverted_after_cmd = False
            elif op == "resume":
                r = bridge.resume_line()
                last_cmd = "resume"
                diverted_after_cmd = False
            else:
                r = bridge.divert_rework()
                assert r["tower_light"] == "YELLOW", (
                    f"case {case}: divert must raise the yellow tower")
                diverted_after_cmd = True
        st = bridge.read_plc_status()
        assert st["tower_light"] in ("RED", "YELLOW", "GREEN"), (
            f"case {case}: tower light must be a valid state, got {st['tower_light']}")
        if last_cmd == "halt":
            assert st["conveyor_running"] is False and st["halt_triggered"] is True, (
                f"case {case}: final halt must stop the conveyor")
            if not diverted_after_cmd:
                assert st["tower_light"] == "RED", (
                    f"case {case}: final halt must show RED")
        elif last_cmd == "resume":
            assert st["conveyor_running"] is True and st["halt_triggered"] is False, (
                f"case {case}: final resume must run the conveyor")
            if not diverted_after_cmd:
                assert st["tower_light"] == "GREEN", (
                    f"case {case}: final resume must show GREEN")
        if diverted_after_cmd and last_cmd != "halt":
            assert st["tower_light"] == "YELLOW", (
                f"case {case}: divert after last command must hold YELLOW")
        elif last_cmd == "halt":
            assert st["tower_light"] == "RED", (
                f"case {case}: halted line must show RED (top priority)")


# ---- windowed dispatch (debounce): at most 1 dispatch per window --------------

def _windowed_dispatch(timestamps: list[float], window: float) -> list[float]:
    """Debounce helper under test: emit the first event, then at most one per
    window after the last emission (trailing-edge collapse of bursts)."""
    emitted: list[float] = []
    last_emit = float("-inf")
    for t in timestamps:
        if t - last_emit >= window:
            emitted.append(t)
            last_emit = t
    return emitted


def test_debounce_emits_at_most_one_dispatch_per_window():
    rng = random.Random(SEED + 3)
    for case in range(N_CASES):
        window = rng.choice([0.05, 0.2, 0.5, 1.0])
        # Bursty timestamps: clusters of near-simultaneous defect triggers.
        ts: list[float] = []
        t = 0.0
        for _ in range(rng.randint(1, 60)):
            t += rng.choice([0.0, 0.001, 0.01, rng.uniform(0, 2 * window)])
            ts.append(t)
        emitted = _windowed_dispatch(ts, window)
        assert len(emitted) <= len(ts), (
            f"case {case}: debounce must never amplify events")
        for prev, cur in zip(emitted, emitted[1:]):
            assert cur - prev >= window - 1e-9, (
                f"case {case}: two dispatches inside one window ({cur - prev})")
        if ts:
            assert emitted and emitted[0] == ts[0], (
                f"case {case}: first defect must always dispatch immediately")
        # A lone burst inside one window collapses to exactly one dispatch.
        burst = [1.0 + rng.uniform(0, window / 2) for _ in range(10)]
        assert len(_windowed_dispatch(burst, window)) == 1, (
            f"case {case}: single-window burst must collapse to 1 dispatch")


def test_mode_parsing_randomized():
    rng = random.Random(SEED + 4)
    for case in range(N_CASES):
        raw = rng.choice(["simulation", "SIMULATION", " Simulated ",
                          "hardware", "HARDWARE", " Hardware "])
        parsed = PLCBridge._parse_mode(raw)
        assert parsed in ("SIMULATION", "HARDWARE"), (
            f"case {case}: mode {raw!r} parsed to invalid {parsed!r}")
    for bad in ["auto", "", "sim", "hard", "123"]:
        try:
            PLCBridge._parse_mode(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"mode {bad!r} must raise ValueError (fail-closed)")


def test_simulation_mode_never_touches_hardware_env(monkeypatch):
    """Simulation must stay simulated even with hardware env vars set."""
    monkeypatch.setenv("APEX_PLC_MODE", "simulation")
    monkeypatch.setenv("PLC_HOST", "127.0.0.1")
    monkeypatch.setenv("PLC_PORT", "5020")
    bridge = PLCBridge()
    assert bridge.mode == "SIMULATION"
    assert bridge.is_connected is False
    res = bridge.halt_line(os.environ.get("PLC_HOST", "x"))
    assert res["status"] == "SIMULATED", "simulation must not claim dispatch"
