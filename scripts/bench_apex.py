#!/usr/bin/env python3
"""Micro-benchmark: ApexInspect IoU / NMS-lite / 3-consecutive debounce math.

Pure stdlib. Optionally uses src.vision.patching IoU if import-clean
(cv2 present); otherwise runs the stdlib baseline labeled as such.
Never fails.

Run: python3 scripts/bench_apex.py   (from ApexInspect-AI/)
"""
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

try:
    from vision.patching import HighResPatchInferencer  # needs cv2

    MODE = "stdlib math (patching import ok, timing pure math for hermeticity)"
except Exception as exc:
    MODE = f"stdlib math baseline (src.vision.patching unavailable: {exc})"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


BOXES = [(10.0, 10.0, 50.0, 50.0), (12.0, 12.0, 52.0, 52.0), (100.0, 100.0, 140.0, 140.0)]
STREAM = [True, True, False, True, True, True, False, True, True, True] * 100


def op_iou():
    return iou(BOXES[0], BOXES[1])


def op_debounce():
    consec = 0
    fired = 0
    for defective in STREAM:
        consec = consec + 1 if defective else 0
        if consec >= 3:
            fired += 1
            consec = 0  # ticket opened, streak reset
    return fired


def timed(fn, n):
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return n, statistics.fmean(samples), samples[min(n - 1, int(n * 0.95))]


def main():
    print(f"mode: {MODE}")
    print(f"{'op':<18}{'n':>8}{'mean_ms':>12}{'p95_ms':>12}")
    for name, fn, n in (("iou", op_iou, 20_000), ("debounce", op_debounce, 2_000)):
        nn, mean, p95 = timed(fn, n)
        print(f"{name:<18}{nn:>8}{mean:>12.4f}{p95:>12.4f}")


if __name__ == "__main__":
    main()
