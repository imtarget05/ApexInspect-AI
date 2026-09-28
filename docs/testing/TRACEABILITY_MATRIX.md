# Traceability Matrix — ApexInspect-AI

`Requirement → Business Rule → Test Case → Automated Test → Execution Evidence`

| Business invariant | Test Case | Automated test (exact node, on disk) | Source file | Evidence |
|---|---|---|---|---|
| FAIL_CLOSED_QUARANTINE (corrupt/empty/unsupported/ONNX-error → quarantine, never PASS) | APEX-003–007 | `test_missing_model_does_not_return_a_clean_board`, `test_missing_model_renders_unknown_not_pass`, `test_strict_mode_raises_instead_of_returning_no_verdict`, `test_corrupt_model_does_not_report_a_pass`, `test_require_model_raises_when_degraded` | `tests/test_detector_fail_closed.py` | VERIFIED 8 passed 2026-09-27 (was CI-ONLY; cv2 installed this session) |
| CLASSIFY_ACCURACY (good passes, defective quarantined — needs real runtime) | APEX-001, APEX-002, APEX-008 | `test_detector_property.py` (6, green); inference tests RED without runtime | `tests/test_detector_property.py`, `tests/test_vision.py`, `tests/test_patching.py` | BLOCKED → ENV-APEX-003 (onnxruntime uninstallable on cp314) |
| DETECTOR_SANE (good passes, bad quarantined) | APEX-001, APEX-002 | `test_detector_property.py` (6 fns) | `tests/test_detector_property.py` | VERIFIED 6/6 2026-09-27 |
| NO_REPLAY_ON_RECONNECT (no duplicate actuation) | APEX-011–013 | `test_hardware_write_exception_returns_failed_without_simulation_claim`, `test_hardware_stops_remaining_coil_writes_after_rejected_response` | `tests/test_plc_bridge.py` | VERIFIED 2026-09-27 (in 13-run) |
| SIM_ONLY (never touches real PLC) | APEX-014, APEX-015 | `test_plc_bridge_offline_simulation_mode`, `test_simulation_mode_never_connects_or_writes_to_hardware`, `test_hardware_mode_rejects_unavailable_plc` | `tests/test_plc_bridge.py` | VERIFIED 2026-09-27 |
| TAMPER_EVIDENT (edit/delete → detected) | APEX-016–018 | `test_chain_200_seeded_records_verify_passes`, `test_one_byte_tamper_detected_at_exact_index`, `test_truncation_detected` | `tests/test_audit_chain.py` | VERIFIED 2026-09-27 (in 13-run) |
| NO_FLAP (hysteresis 15%/12%) | APEX-019–021 | `test_500_seeded_random_walk_no_flap_inside_band`, `test_exact_fire_and_clear_edges` | `tests/test_alert_hysteresis.py` | VERIFIED 2026-09-27 (in 13-run) |
| APPROVE_ONCE_DISPATCH_ONCE | HITL→PLC | `test_repeated_approval_is_rejected_without_a_second_plc_dispatch`, `test_replay_same_key_returns_stored_response_without_second_dispatch` | `tests/test_approval_safety.py` | UNVERIFIED here (CI-ONLY, no sqlalchemy) |
