# Test Plan — ApexInspect-AI (Vision + PLC)

**Contract:** [`docs/qa/QA_ACCEPTANCE.md`](../../docs/qa/QA_ACCEPTANCE.md).
**Case IDs:** `APEX-001` → `APEX-021` in `TEST_CASES.md`.
**Runner:** `pytest`. On disk: 28 files, **173 `def test`** (enumerated 2026-09-27; map first — do not duplicate).

## Scope

ONNX edge inference (good/defective/corrupt/empty/unsupported inputs), model-missing
safety, inference SLA `<50ms`, soak (1000 images), Modbus TCP PLC (normal/timeout/
disconnect/reconnect, simulation mode), audit hash chain (verify/tamper/delete),
alert hysteresis.

## Levels

| Level | What | Where |
|---|---|---|
| Unit | fail-closed branches, hysteresis math, audit-chain verify on fixtures | `test_detector_fail_closed.py` (9), `test_alert_hysteresis.py` (3), `test_audit_chain.py` (4) |
| Property | detector invariants, blur/brightness evidence-gap thresholds | `test_detector_property.py` (6), `test_quality.py` (9) |
| Integration | inference → quarantine decision; PLC command → register assertion (simulator); tamper → detected | `test_vision.py` (8), `test_plc_bridge.py` (7) |
| Adversarial | corrupt/empty/unsupported images, ONNX exception, model missing, mid-command disconnect | `test_detector_fail_closed.py`, `test_vision_adversarial` path |
| Race | repeated approval → single PLC dispatch; replay same key → stored response | `test_approval_safety.py` (10) |
| Soak / Performance | 1000-image run (no leak/crash), p95 inference latency vs 50ms SLA | `scripts/bench_apex.py` (43.36 ms/frame last-known, UNVERIFIED here) |
| E2E | defective PCBA → quarantine + audit record + alert per hysteresis rule | line simulator |

## Environments

| Env | Command | Scope |
|---|---|---|
| Offline (dep-light) | `/tmp/factorygen/bin/python -m pytest -q tests/test_detector_property.py tests/test_audit_chain.py tests/test_alert_hysteresis.py tests/test_plc_bridge.py` | **19 passed, 1 skipped** 2026-09-27 |
| Offline (blocked) | `test_detector_fail_closed.py` → `ModuleNotFoundError: cv2`; `test_approval_safety.py` → `ModuleNotFoundError: sqlalchemy` | CI-ONLY (verified 2026-09-27) |
| CI | `pytest` after requirements install | full 173-test suite + 6 Phase-2 (UNVERIFIED here) |
| Live-infra | `APEX_PLC_MODE=simulation` always in tests | **Real PLC is never touched** (APEX-014); wrong-mode config fails at startup (APEX-015) |

## Entry / exit criteria

- Entry: model artifact + threshold config pinned.
- Exit: P0 100% PASS; `UNKNOWN/ERROR/TIMEOUT = QUARANTINE` proven at unit + integration + E2E.

## Invariant under test

```text
UNKNOWN / ERROR / TIMEOUT = QUARANTINE   (never ERROR = PASS)
```
