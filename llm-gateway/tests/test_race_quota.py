"""Phase-2 hard tests: threading races + pure-unit determinism probes.

Pure unit only: stdlib threading/random, no TestClient, no network. The
global quota/circuit state is reset before and after every test so the rest
of the suite is unaffected.
"""
from __future__ import annotations

import copy
import random
import string
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server  # noqa: E402


@pytest.fixture
def clean_state(monkeypatch, tmp_path, request):
    """Isolate global server state; always point the log at tmp_path."""
    monkeypatch.setenv("LLM_LOG_PATH", str(tmp_path / "pytest-race-telemetry.jsonl"))
    server._reset_circuit_state()
    server._reset_quota_state()
    server._load_config()

    def teardown():
        server._reset_circuit_state()
        server._reset_quota_state()
        server.telemetry.close()
        server._load_config()  # env restored by monkeypatch by now

    request.addfinalizer(teardown)
    return tmp_path


# ------------------------------------------------------------------ (a) quota


@pytest.mark.race
def test_quota_add_thread_race_exact_total(clean_state):
    """16 threads x 200 _quota_add(1) must total exactly 3200.

    NOTE (report, not fix): _quota_add is a read-modify-write
    (``_quota_usage[key] = _quota_usage.get(key, 0) + n``) with no lock, so
    under a free-threaded interpreter this could lose updates. Under the
    GIL the bytecode window is small enough that 3200/3200 holds
    empirically (20/20 barrier-aligned trials during development).
    """
    project = "RaceQuotaProj"
    n_threads, n_increments = 16, 200
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait()  # maximise contention: release all threads at once
        for _ in range(n_increments):
            server._quota_add(project, 1)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "worker thread hung"

    key = (project, server._quota_day())
    assert server._quota_usage.get(key) == n_threads * n_increments


# ------------------------------------------------------------------ (b) breaker


@pytest.mark.race
def test_circuit_breaker_trips_exactly_at_threshold(clean_state):
    """8 threads racing _cb_record_failure trip open with failures == 8.

    Threshold is set to the thread count so any lost update would leave the
    breaker closed (failures < 8) and fail this test. Same GIL note as the
    quota test: the increment is an unlocked read-modify-write.
    """
    monkeypatch_threshold = 8
    server.CB_FAILURE_THRESHOLD = monkeypatch_threshold
    server.CB_OPEN_SECONDS = 30.0
    base = "http://race.local/v1"
    barrier = threading.Barrier(monkeypatch_threshold)

    def worker():
        barrier.wait()
        server._cb_record_failure(base)

    threads = [threading.Thread(target=worker) for _ in range(monkeypatch_threshold)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "worker thread hung"

    entry = server._cb_state.get(base)
    assert entry is not None
    assert entry["failures"] == monkeypatch_threshold
    assert entry["state"] == "open"


# ------------------------------------------------------------------ (c) PII idempotency


PII_BODIES = [
    {"messages": [{"role": "user", "content": "gọi 0901234567 cccd 001234567890 nhé"}]},
    {"messages": [{"role": "user", "content": "tel +84 901 234 567, cmnd 123456789"}]},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "sđt 0912.345.678"}]}]},
    {"input": "my number 0987654321"},
    {"input": ["call 0331112223", "plain text"]},
    {"messages": [{"role": "user", "content": f"already {server.PII_TOKEN} here"}]},
    {"messages": [{"role": "user", "content": "no pii at all, just hello"}]},
    {"model": "m"},
    {},
]


@pytest.mark.race
def test_pii_mask_body_idempotent(clean_state):
    """mask(mask(x)) == mask(x) for every crafted body; no caller mutation."""
    for body in PII_BODIES:
        snapshot = copy.deepcopy(body)
        masked_once, n1 = server._pii_mask_body(body)
        assert body == snapshot, "masking must never mutate the caller's dict"
        masked_twice, n2 = server._pii_mask_body(masked_once)
        assert masked_twice == masked_once, f"not idempotent for {body!r}"
        if n1:
            assert n2 == 0, f"second pass should find nothing left to mask: {body!r}"


# ------------------------------------------------------------------ (d) alias fuzz


@pytest.mark.race
def test_resolve_model_alias_fuzz_never_raises(clean_state):
    """500 seeded-random model strings: never raise, always non-empty str."""
    rng = random.Random(20260927)
    alphabet = (
        string.ascii_letters
        + string.digits
        + " _-./:@#\t"
        + "àáâäçèéê HồChíMinh	GPT"
        + "🤖🔥"
    )
    known = ["", " ", "auto", "AUTO", "Fast", "fast", "reasoning", "embed",
             "vision", "../etc/passwd", "a" * 500, "\n", "null", "none",
             "deepseek/deepseek-r1-0528-qwen3-8b", "text-embedding-nomic-embed-text-v1.5"]
    tasks = [None, "", "auto", "fast", "embed", "reasoning", "bogus-task"]
    for i in range(500):
        if i < len(known):
            model = known[i]
        else:
            model = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        task = tasks[i % len(tasks)]
        result = server.resolve_model_alias(model, task=task)  # must never raise
        assert isinstance(result, str), f"non-str for {model!r}"
        assert result != "", f"empty resolution for {model!r}"
