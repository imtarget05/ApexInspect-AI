# Test Execution Report — ApexInspect-AI

Cases: [`TEST_CASES.md`](TEST_CASES.md) (`APEX-001`→`APEX-021`).
Evidence dir: [`evidence/`](evidence/). Session date: 2026-09-27 UTC.
Runner for offline rows: `/tmp/factorygen/bin/python -m pytest -q`.

| Date (UTC) | Command | Scope | Result | Verdict | Evidence |
|---|---|---|---|---|---|
| 2026-09-27 | `pytest -q tests/test_detector_property.py` | APEX-001 (+002 property slice) | **6 passed** (1.90 s) | VERIFIED ✅ | `evidence/2026-09-27-dep-light.log` |
| 2026-09-27 | `pytest -q tests/test_audit_chain.py tests/test_alert_hysteresis.py tests/test_plc_bridge.py` | APEX-010–021 (excl. soak/bench) | **13 passed, 1 skipped** (18.11 s) | VERIFIED ✅ | same log |
| 2026-09-27 | `pytest -q tests/test_detector_fail_closed.py` | APEX-003–007 | collection ERROR: `ModuleNotFoundError: cv2` | ENV-BLOCKED (CI-ONLY) | same log |
| 2026-09-27 | `pytest --collect-only -q tests/test_approval_safety.py` | HITL→PLC dispatch (10 defs) | collection ERROR: `ModuleNotFoundError: sqlalchemy` | ENV-BLOCKED (CI-ONLY) | same log |
| — | `pytest` (full suite, 173 defs + 6 Phase-2) | APEX-001→021 | UNVERIFIED (HARD_TEST_REPORT.md §III) | SUPERSEDED by row below | rerun in CI |
| — | `scripts/bench_apex.py` | APEX-008/009 | UNVERIFIED — last-known 43.36 ms/frame (NOT A GATE) | UNVERIFIED | — |
| 2026-09-27 | `ApexInspect-AI/.venv/bin/python -m pytest -q -rs` (repo `.venv` + opencv 4.12.0.88) | full suite incl. 8 previously cv2-gated files | **168 passed, 2 failed, 2 skipped** (61.87 s) | VERIFIED ✅ with 2 env-rooted reds (ENV-APEX-003) | `evidence/2026-09-27-pytest-full.log` (+ `evidence/2026-09-27-pytest-vision.log` for the 8-file slice: 45 passed, same 2 reds) |
| — | `scripts/bench_apex.py` | APEX-008/009 | UNVERIFIED — last-known 43.36 ms/frame (NOT A GATE) | UNVERIFIED | — |

## How to record a run

1. Run the suite (see `TEST_PLAN.md`; PLC simulator only, never the real line).
2. Save raw output under `evidence/YYYY-MM-DD-<scope>.log`.
3. Fill one row above; update `Status` in `TEST_CASES.md`.
4. Any FAIL/FLAKY gets an entry in `DEFECT_REPORT.md` before the run counts as reviewed.
