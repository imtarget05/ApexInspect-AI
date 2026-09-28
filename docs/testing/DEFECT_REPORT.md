# Defect Report — ApexInspect-AI

| ID | Severity | Title | Repro | Evidence | Status |
|---|---|---|---|---|---|
| ENV-APEX-001 | S4 (env-only) | `test_detector_fail_closed.py` (APEX-003–007) not collectable offline: `ModuleNotFoundError: cv2` via `src/vision/detector.py:3` | 2026-09-27: `pytest -q tests/test_detector_fail_closed.py` → collection ERROR | `evidence/2026-09-27-dep-light.log` | OPEN (CI installs `requirements.txt`; not an app defect) |
| ENV-APEX-002 | S4 (env-only) | `test_approval_safety.py` (10 defs) not collectable offline: `ModuleNotFoundError: sqlalchemy` | 2026-09-27: `pytest --collect-only` → collection ERROR | same log | OPEN (same as above) |

Session failures in app logic: **none** (19 passed + 1 skipped on the dep-light slice).
Historical defects with on-disk evidence in this repo: **none verified** — none recorded.

## Full-run findings 2026-09-27 (168 passed, 2 failed, 2 skipped)

| ID | Severity | Title | Repro | Evidence | Status |
|---|---|---|---|---|---|
| ENV-APEX-001 | S4 (env-only) | cv2 block — SUPERSEDED: `opencv-python-headless==4.12.0.88` installed into repo `.venv`; all 8 gated files now collect and run (fail_closed 8 passed) | install + full run 2026-09-27 | `evidence/2026-09-27-pytest-full.log` | SUPERSEDED (env fixed; side effect: numpy 2.5.3→2.2.6, no regression observed) |
| ENV-APEX-002 | S4 (env-only) | sqlalchemy block — SUPERSEDED: `test_approval_safety.py` (10) collects and passes in the full run | same log | same log | SUPERSEDED |
| ENV-APEX-003 | P0-env-BLOCKED | `onnxruntime` uninstallable on this host's Python 3.14 (no cp314 wheel) → any test needing real inference RED: `test_detector_inference_latency_under_threshold` (TypeError on `latency_ms=None`), `test_infer_high_res_end_to_end`. The fail-closed behavior itself is correct (`ModelUnavailableError`, "No defect verdict is possible") | full run: 2 failed, both `ModuleNotFoundError: onnxruntime` at root | same log | BLOCKED with cause (APEX-001/002/008 need runtime + committed model) |
| GAP-APEX-001 | P1 (coverage gap) | No soak test (APEX-009): no 1000-image / leak run on disk | grep soak in `tests/` → no test (only dataset-split "leak" asserts) | `TEST_CASES.md` APEX-009 | OPEN |

## Lifecycle

`OPEN → FIXED` (with re-test evidence) or `OPEN → MITIGATED` (workaround +
root-cause tracking ID) or `→ WONTFIX` (justification required for P0/P1).
Every defect links the failing case ID from `TEST_CASES.md`.
