# Test Cases — ApexInspect-AI (Vision + PLC)

**Plan:** [`TEST_PLAN.md`](TEST_PLAN.md). **Contract:** `docs/qa/QA_ACCEPTANCE.md`.
On-disk: 28 files / 173 `def test`. ✅ = ran green 2026-09-27; CI-ONLY = needs
`cv2`/`sqlalchemy`/models absent offline.

| ID | file::case (on disk) | Invariant asserted | Runnable offline? |
|---|---|---|---|
| APEX-001 | `test_detector_property.py` (6) ✅ | good PCBA passes | ✅ 6 passed |
| APEX-002 | `test_quality.py` (9: `test_reject_blurred_frame`, `test_accept_good_frame`, …) | defective PCBA quarantined | CI |
| APEX-003 | `test_detector_fail_closed.py::test_corrupt_model_does_not_report_a_pass` | corrupt → QUARANTINE | CI-ONLY (no cv2) |
| APEX-004 | `test_detector_fail_closed.py::test_missing_model_renders_unknown_not_pass` | empty → QUARANTINE | CI-ONLY |
| APEX-005 | `test_detector_fail_closed.py` unsupported-format path | fail closed | CI-ONLY |
| APEX-006 | `test_detector_fail_closed.py::test_strict_mode_raises_instead_of_returning_no_verdict` | ONNX error → never PASS | CI-ONLY |
| APEX-007 | `test_detector_fail_closed.py::test_missing_model_does_not_return_a_clean_board`, `::test_require_model_raises_when_degraded` | line never runs unsafe | CI-ONLY |
| APEX-008 | `scripts/bench_apex.py` (43.36 ms/frame last-known) | p95 `<50ms` | UNVERIFIED (NOT A GATE) |
| APEX-009 | soak script (no test file) | 1000 images, no leak | UNVERIFIED |
| APEX-010 | `test_plc_bridge.py::test_plc_bridge_with_virtual_modbus_server` ✅ | correct coil/register write | ✅ (13 passed + 1 skip w/ audit+alert) |
| APEX-011 | `test_plc_bridge.py::test_hardware_write_exception_returns_failed_without_simulation_claim` ✅ | timeout ≠ success | ✅ |
| APEX-012 | `test_plc_bridge.py::test_hardware_stops_remaining_coil_writes_after_rejected_response` ✅ | fail safe mid-command | ✅ |
| APEX-013 | `test_plc_bridge.py` reconnect path ✅ | no duplicate actuator command | ✅ |
| APEX-014 | `test_plc_bridge.py::test_plc_bridge_offline_simulation_mode`, `::test_simulation_mode_never_connects_or_writes_to_hardware` ✅ | never touches real PLC | ✅ |
| APEX-015 | `test_plc_bridge.py::test_hardware_mode_rejects_unavailable_plc` ✅ | bad config fails at startup | ✅ |
| APEX-016 | `test_audit_chain.py::test_chain_200_seeded_records_verify_passes`, `::test_empty_log_verify_true` ✅ | chain verifies 100% | ✅ |
| APEX-017 | `test_audit_chain.py::test_one_byte_tamper_detected_at_exact_index` ✅ | 1-byte edit → CHAIN BROKEN | ✅ |
| APEX-018 | `test_audit_chain.py::test_truncation_detected` ✅ | delete → broken link | ✅ |
| APEX-019 | `test_alert_hysteresis.py::test_500_seeded_random_walk_no_flap_inside_band` ✅ | no flapping 14.9–15.1% | ✅ |
| APEX-020 | `test_alert_hysteresis.py::test_exact_fire_and_clear_edges` ✅ | fires over 15% | ✅ |
| APEX-021 | `test_alert_hysteresis.py::test_constructor_rejects_off_ge_on` + clear edge ✅ | clears below 12% | ✅ |

## Supplementary on-disk coverage

| File | `def test` | Notes |
|---|---|---|
| `test_approval_safety.py` 10 (`test_duplicate_pending_ticket_is_not_created`, `test_repeated_approval_is_rejected_without_a_second_plc_dispatch`, `test_replay_same_key_returns_stored_response_without_second_dispatch`, …) | 10 | HITL→PLC dispatch safety, CI-ONLY (no sqlalchemy) |
| `test_api_auth_hardening.py` 16, `test_security_audit.py` 3, `test_llm_guard.py` 6, `test_db_guard.py` 4, `test_db_isolation.py` 2, `test_health_contract.py` 2 | 33 | auth/guards/health |
| `test_agent.py` 6, `test_eval_harness.py` 14, `test_incident_memory.py` 19, `test_rca_groq_vcr.py` 1, `test_backend.py` 9, `test_dashboard.py` 6, `test_ingest.py` 6, `test_quality.py` 9, `test_stream.py` 2, `test_vision.py` 8, `test_sample_gallery.py` 5, `test_prepare_dataset.py` 5, `test_training_yaml.py` 4, `test_workflow_tables.py` 1, `test_patching.py` 4, `test_deploy_smoke.py` 2 | 91 | agent/eval/vision/data/ops |

**Invariant:** `UNKNOWN / ERROR / TIMEOUT = QUARANTINE` — never `ERROR = PASS`.

## Full-run verdicts 2026-09-27 (repo `.venv`; `opencv-python-headless==4.12.0.88` installed this session)

Full suite: **168 passed, 2 failed, 2 skipped** — `evidence/2026-09-27-pytest-full.log`.
(The 8 cv2-gated files previously CI-ONLY now execute; `test_approval_safety.py` also
collects — sqlalchemy present. Side effect to watch: opencv install downgraded
numpy 2.5.3 → 2.2.6; no regression observed in this run.)

- APEX-001 → PARTIAL (property suite green; true classification accuracy needs ONNX
  runtime + committed model — both absent, see below).
- APEX-002 → PASS with note (`test_quality.py` 9 green on synthetic frames; real-model
  accuracy BLOCKED same as APEX-001).
- APEX-003–007 → PASS (`test_detector_fail_closed.py` 8 passed, 1 skipped: missing/
  corrupt model → unknown/raise, never PASS; strict mode raises).
- APEX-008 → BLOCKED (test `test_detector_inference_latency_under_threshold` exists
  with 120 ms budget but RED without runtime — `latency_ms=None` → TypeError).
- APEX-009 → GAP P1 (no soak test on disk — GAP-APEX-001).
- APEX-010–021 → PASS (unchanged, re-verified in full run).
- The 2 red tests (`...latency_under_threshold`, `test_infer_high_res_end_to_end`) share
  one cause: `ModuleNotFoundError: onnxruntime` (no cp314 wheel) → ENV-APEX-003.
- Skips: committed-model (needs ONNX file), VCR (vcrpy absent) — both reasoned.

**Gate verdict: NOT QA READY** — P0 classification accuracy + latency BLOCKED on
onnxruntime env (ENV-APEX-003); P1 soak gap (GAP-APEX-001). Fail-closed direction
(the safety-critical half) is fully green.
