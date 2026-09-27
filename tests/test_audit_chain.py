"""Audit hash chain tests: stdlib only, no repo imports beyond the module."""

import importlib.util
import json
import os
import random


def _load_module(name, relpath):
    path = os.path.join(os.path.dirname(__file__), "..", *relpath.split("/"))
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_chain = _load_module("apex_audit_chain", "src/agent/audit_chain.py")
append, verify = _chain.append, _chain.verify


def _seeded_records(n=200, seed=20260927):
    rng = random.Random(seed)
    edge_strings = [
        "",
        " ",
        "\n\t\r",
        "café ☕ ñ 中文 日本語 한국어 emoji 🎛️🏭",
        "e" + "\u0301",  # combining accent
        "null\x00byte",
        '"quotes" \\ backslash',
        "x" * 5000,
        "line1\nline2\nline3",
        "🔒" * 500,
        "\U0001f600\U0001f4a9",
        "  leading/trailing  ",
        "{not json} [1,2",
        "NaN Infinity -Infinity",
    ]
    records = []
    for i in range(n):
        kind = rng.random()
        if kind < 0.35:
            records.append({
                "seq": i,
                "action": rng.choice(["TRIGGER_CONSECUTIVE_DEFECTS", "TRIGGER_YIELD_DRIFT",
                                      "APPROVE_ACTION", "REJECT_ACTION", "RESUME_LINE"]),
                "operator_id": rng.choice(["AI_INCIDENT_AGENT", "supervisor", "opérätor-☕", ""]),
                "line_id": f"L{rng.randint(1, 5)}",
                "ticket_id": f"TICK-{i:04d}-{rng.randint(0, 9999):04d}",
                "defect_ratio": rng.random() * 0.5,
                "note": rng.choice(edge_strings),
            })
        elif kind < 0.65:
            records.append({
                "seq": i,
                "nested": {"a": [1, 2.5, None, True, {"k": rng.choice(edge_strings)}],
                           "unicode": rng.choice(edge_strings)},
                "counts": list(range(rng.randint(0, 10))),
                "flag": rng.choice([True, False, None]),
                "value": rng.choice([0, -1, 1e308, 1e-308, 3.14159]),
            })
        else:
            # Adversarial-but-legal JSON payloads
            records.append({
                "seq": i,
                rng.choice(edge_strings) or "k": rng.choice(edge_strings),
                "num_key_1": rng.randint(-10**12, 10**12),
                "empty": {} if rng.random() < 0.5 else [],
            })
    return records


def test_chain_200_seeded_records_verify_passes(tmp_path):
    log = str(tmp_path / "audit.chain.jsonl")
    records = _seeded_records()
    for rec in records:
        entry = append(rec, log)
        assert entry["prev_hash"]
        assert len(entry["entry_hash"]) == 64
    ok, bad = verify(log)
    assert ok is True and bad is None
    # Chain linkage spot-check
    lines = open(log, encoding="utf-8").read().strip().split("\n")
    assert len(lines) == 200
    prev = "GENESIS"
    for idx, line in enumerate(lines):
        e = json.loads(line)
        assert e["index"] == idx
        assert e["prev_hash"] == prev
        prev = e["entry_hash"]


def test_one_byte_tamper_detected_at_exact_index(tmp_path):
    rng = random.Random(7)
    log = str(tmp_path / "audit.chain.jsonl")
    n = 200
    for rec in _seeded_records(n, seed=99):
        append(rec, log)
    raw = open(log, "rb").read()
    # Pick a random entry line and flip one byte inside its body.
    lines = raw.split(b"\n")
    assert lines[-1] == b""
    lines = lines[:-1]
    target = rng.randrange(n)
    line = bytearray(lines[target])
    pos = rng.randrange(len(line))
    orig = line[pos]
    line[pos] = (orig + 1) % 256 if orig != 0x0A else 0x20
    lines[target] = bytes(line)
    open(log, "wb").write(b"\n".join(lines) + b"\n")
    ok, bad = verify(log)
    assert ok is False
    assert bad == target, f"expected tamper at {target}, got {bad}"


def test_truncation_detected(tmp_path):
    log = str(tmp_path / "audit.chain.jsonl")
    for rec in _seeded_records(20, seed=5):
        append(rec, log)
    raw = open(log, "rb").read()
    # Chop 10 bytes off the end -> partial last line.
    open(log, "wb").write(raw[:-10])
    ok, bad = verify(log)
    assert ok is False
    assert bad == 19


def test_empty_log_verify_true(tmp_path):
    log = str(tmp_path / "empty.chain.jsonl")
    open(log, "w", encoding="utf-8").close()
    assert verify(log) == (True, None)
    assert verify(str(tmp_path / "missing.chain.jsonl")) == (True, None)
