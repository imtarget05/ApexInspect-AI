"""Hysteresis alert tests: stdlib only."""

import importlib.util
import os
import random

import pytest


def _load_module(name, relpath):
    path = os.path.join(os.path.dirname(__file__), "..", *relpath.split("/"))
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HysteresisAlert = _load_module("apex_alert_hysteresis", "src/backend/alert_hysteresis.py").HysteresisAlert


def test_constructor_rejects_off_ge_on():
    with pytest.raises(ValueError):
        HysteresisAlert(on_threshold=0.12, off_threshold=0.12)
    with pytest.raises(ValueError):
        HysteresisAlert(on_threshold=0.10, off_threshold=0.12)


def test_exact_fire_and_clear_edges():
    a = HysteresisAlert(on_threshold=0.15, off_threshold=0.12)
    assert a.state is False
    assert a.update(0.149999) is False
    assert a.update(0.15) is True  # fires at >= on
    assert a.state is True
    assert a.update(0.12) is True  # holds at exactly off (clears only when < off)
    assert a.update(0.119999) is False  # clears below off
    assert a.state is False


def test_500_seeded_random_walk_no_flap_inside_band():
    rng = random.Random(20260927)
    a = HysteresisAlert(on_threshold=0.15, off_threshold=0.12)
    rate = 0.135
    transitions_inside = 0
    prev = a.state
    for _ in range(500):
        # Random walk clamped around the band so many samples fall inside (12%, 15%).
        rate += rng.uniform(-0.01, 0.01)
        rate = min(0.20, max(0.07, rate))
        # Bias samples into the band to stress the dead zone.
        if rng.random() < 0.6:
            rate = rng.uniform(0.121, 0.149)
        cur = a.update(rate)
        if 0.12 <= rate < 0.15 and cur != prev:
            transitions_inside += 1
        # Outside the band, behaviour must be exact.
        if rate >= 0.15:
            assert cur is True
        if rate < 0.12:
            assert cur is False
        prev = cur
    assert transitions_inside == 0, f"flapped {transitions_inside}x inside the band"
