# Recruiter Evidence — ApexInspect AI

Every line below was produced by running the code in this repository during a single
audit session. Nothing here is inferred from the README, docstrings, or commit
messages. Where the repository's own documentation overstates what the code does,
this document says so.

## How this was verified

- Repository: `imtarget05/ApexInspect-AI` at commit `aee1d6e` ("chore: trigger CI verification").
- Interpreter: CPython 3.11.0, Windows, 8 logical CPUs.
- Environment built from scratch: `uv venv --python 3.11 .venv` then
  `uv pip install --python .venv\Scripts\python.exe -r requirements.txt`
  -> **Installed 146 packages in 21.71s** (onnxruntime 1.30.0, opencv 5.0.0,
  numpy 2.4.6, pymodbus 3.15.0, langgraph 1.2.12, fastapi 0.141.1,
  sqlalchemy 2.1.1, ultralytics 8.4.153, torch 2.14.0). The full
  `requirements.txt` installed cleanly; no fallback to
  `requirements-runtime.txt` was needed.
- `pytest` is not in `requirements.txt`; like CI, it was installed separately
  (`uv pip install pytest pytest-vcr`).
- No source file was modified. `git status --porcelain` after the audit shows
  only the pre-existing untracked `docs/REPAIR-PLAN.md`. The benchmark was run in
  a throwaway copy of the tree because `scripts/benchmark_inference.py:77`
  overwrites the committed `benchmark_results.json`.

Status vocabulary used throughout: `VERIFIED`, `PARTIALLY VERIFIED`,
`NOT VERIFIED`, `FAILED`, `NOT APPLICABLE`.

---

## Trained a 6-class YOLOv8n PCB-defect detector (fine-tuned on a 693-image Kaggle corpus) and deployed it as a committed 12 MB ONNX graph that loads and executes on CPU via ONNX Runtime

- **STATUS**: PARTIALLY VERIFIED
- **IMPLEMENTATION**: `PCBDefectDetector._initialize_engine` (`src/vision/detector.py:86-114`) opens an `onnxruntime.InferenceSession` with `GraphOptimizationLevel.ORT_ENABLE_ALL` and `providers=["CPUExecutionProvider"]`, sizing `intra_op_num_threads` to `max(2, cpu_count // 2)`. `resolve_model_path` (`src/vision/detector.py:14-42`) resolves the canonical bundle in priority order arg -> `$MODEL_PATH` -> `models/yolov8n_pcb_defect.onnx`. `preprocess` (`src/vision/detector.py:155-167`) runs CLAHE on the Y channel, Ultralytics-style letterbox to 640x640 with 114-grey padding, then `1/255` normalisation with `swapRB=True`. `_postprocess_yolov8` (`src/vision/detector.py:169-245`) decodes the raw `[1, 10, 8400]` tensor, un-letterboxes boxes, and applies `cv2.dnn.NMSBoxes` at IoU 0.45. `CLASS_ALIASES` (`src/vision/detector.py:55-57`) maps the raw training label `short` -> `short_circuit`. Training provenance is recorded in `models/MODEL_CARD.md:3-23` (Colab T4, 50 epochs, imgsz 640, batch 16, AdamW lr0 0.001, Kaggle `akhatova/pcb-defects`).
- **FILE**: `src/vision/detector.py`, `models/yolov8n_pcb_defect.onnx` (12,269,712 bytes), `models/MODEL_CARD.md`, `models/train.py`, `notebooks/train_pcb_defect_yolo.ipynb`, `models/training_data.yaml`
- **TEST**: `tests/test_vision.py:87 test_detector_resolves_canonical_model_and_reads_env_threshold` asserts `model_path` ends with `models/yolov8n_pcb_defect.onnx` and, when ONNX is live, that the session reports `input_shape == [1, 3, 640, 640]` and `output_shape == [1, 10, 8400]`. `tests/test_vision.py:100 test_detector_letterbox_preserves_aspect_ratio_and_pads_grey` asserts a 1000x2000 image letterboxes to 640x640 at `r == 640/2000.0`, `dw == 0.0`, `dh == 160.0`, and that the padding pixel equals 114. `tests/test_vision.py:95 test_detector_normalizes_training_label_short` asserts `short -> short_circuit`. `tests/test_vision.py:81 test_canonical_onnx_model_is_deployed` is a **file-size assertion only** (`exists` and `> 1,000,000` bytes) — it does not load the graph.
- **RUNTIME EVIDENCE**: I loaded the committed bundle and ran inference on all 13 committed frames in `data/sample_pcbs/`. Command: `.\.venv\Scripts\python.exe` on a script calling `PCBDefectDetector()` then `det.infer(cv2.imread(f))` 10x per frame after 3 warmups. Result:
  - `use_onnx = True`; session input `images` `[1, 3, 640, 640] float32`, output `output0` `[1, 10, 8400] float32`. The graph really does execute.
  - **Measured latency on real frames: mean 70.44 ms**, per-frame averages spanning 61.90 ms (`mouse_bite_2.jpg`) to 87.52 ms (`missing_hole_1.jpg`); 130 timed inferences total. Roughly 14 FPS single-stream on this 8-core box.
  - **Predicted classes: NONE. Zero detections on all 13 real frames** at the default `CONFIDENCE_THRESHOLD=0.50`. I inspected the raw tensor: the maximum class score across all 8400 anchors is **0.0025-0.0029 on every frame**, and the arg-max class is effectively arbitrary (`mouse_bite` on a `missing_hole` frame, `spur` on a `mouse_bite` frame, `open_circuit` on a `spurious_copper` frame). Lowering the threshold to 0.10 still produced zero detections.
  - Control experiment: feeding an all-zeros tensor returned max score **0.002379**, versus 0.002655 for the real image. A blank input and a real photo score the same, so the committed graph emits no discriminative signal on this imagery. This is a property of the weights, not of the preprocessing — I A/B'd four preprocessing variants (repo pipeline with CLAHE, plain letterbox, plain stretch, uniform grey) and the max score stayed in 0.0024-0.0027 in all cases.

  **What this means for a CV claim:** the ONNX inference *path* is real, loadable, and fast enough, and committing the weights with a provenance card is genuinely good practice. The claim that the model *detects the six defect classes* is **NOT VERIFIED** and, on the evidence above, is not supported at the default threshold. `models/MODEL_CARD.md:43-44` reports "2 of 15 candidate boards rejected by QC (13% miss rate)", which implies 13/15 boards produced detections; I could not reproduce that, and `scripts/build_sample_gallery.py:191-205` (`_qc_ok`) would reject every candidate against this model, so the committed gallery could not have been produced by that verified path at threshold 0.50.

---

## Built a reproducible inference-latency benchmark harness with an automated pass/fail target gate

- **STATUS**: VERIFIED (harness runs and reports honest numbers) / the 35 ms target itself is NOT VERIFIED
- **IMPLEMENTATION**: `run_benchmark` (`scripts/benchmark_inference.py:25-62`) does 5 warmup + 20 measured inferences through the full production path (quality gate -> preprocess -> ONNX -> NMS) and emits avg/min/max latency and derived FPS as JSON. Line 19 sets `ALLOW_GT_FALLBACK=true` for the whole run. `scripts/check_benchmark_target.py:63-103` re-runs the benchmark, compares `avg_latency_ms` against a 35 ms target, and returns exit code 1 on failure so it can block a PR. The committed result lives in `benchmark_results.json`.
- **FILE**: `scripts/benchmark_inference.py`, `scripts/check_benchmark_target.py`, `benchmark_results.json`
- **TEST**: **NOT VERIFIED** — there is no test that executes either script. `tests/` never references `benchmark_inference` or `check_benchmark_target`. The only related test is `tests/test_vision.py:56 test_detector_inference_latency_under_threshold`, which measures latency in-process with a <= 120 ms assertion but does not touch the harness.
- **RUNTIME EVIDENCE**: `python scripts\benchmark_inference.py` (run in a throwaway copy of the tree so the committed JSON stayed untouched). Real result:
  ```
  avg_latency_ms : 86.2      min_latency_ms : 68.62
  max_latency_ms : 127.12    fps            : 11.6
  n_warmup : 5               n_measure      : 20
  onnx_runtime : 1.30.0      hardware       : "linux"
  ```
  The committed `benchmark_results.json` claims **43.36 ms / 23.1 FPS**. My number on this 8-core Windows host is **2x worse**. `models/MODEL_CARD.md:22` states a target of "<= 35 ms (~28 FPS, NFR >= 25 FPS)"; I did not meet it, and `check_benchmark_target.py` would exit 1 here.

  **The frames are synthetic, and that changes what the number means.** Line 60 of the script says so itself: "Synthetic PCB frames via PCBCameraSimulator". They are procedurally drawn cartoon boards from `src/vision/simulator.py`, not the real Kaggle captures. And because `ALLOW_GT_FALLBACK=true` (line 19), the "detections" these frames produce are *injected ground truth*, not model output. So the harness measures a real and honest **compute** number, but it is not evidence of detection quality. The `hardware` field is also cosmetic: line 57 hardcodes `"Apple M-series (local macOS)"` merely because `sys.platform == "darwin"`, which is why my Windows run printed `"linux"`.

---

## Implemented a production frame-quality gate that rejects blurred, dark, and low-resolution captures before they reach the network

- **STATUS**: VERIFIED
- **IMPLEMENTATION**: `check_frame` (`src/vision/quality.py:37-65`) applies three ordered checks and returns a verdict dict with `ok`, `reason`, `laplacian_var`, `native_laplacian_var`, `brightness`, `width`, `height`: (1) **resolution** — rejects if `w < 640 or h < 640`; (2) **blur** — Laplacian variance on a 640x640 `INTER_AREA` downscale must be >= `BLUR_THRESHOLD = 30.0`; (3) **brightness** — mean must fall in `[BRIGHT_MIN=40.0, BRIGHT_MAX=200.0]`. The normalisation to 640px before measuring blur is deliberate and documented at `src/vision/quality.py:6-15`: Laplacian variance is resolution-dependent, and a native-resolution threshold scored a sharp 2240x2016 board at 75.6 and rejected every production frame. All three thresholds are env-overridable per camera via `QUALITY_BLUR_THRESHOLD` / `QUALITY_BRIGHT_MIN` / `QUALITY_BRIGHT_MAX` (`_env_float`, `src/vision/quality.py:26-34`) with no rebuild. The gate is enforced on the real inference path at `src/vision/detector.py:261-266`, before `session.run`.
- **FILE**: `src/vision/quality.py`, `src/vision/detector.py:261-266`
- **TEST**: `tests/test_quality.py` (9 tests, all passing). `test_reject_blurred_frame:18` asserts a Gaussian-blurred frame is rejected with `reason == "blur"`. `test_verdict_is_resolution_invariant:34` asserts a sharp 640px frame *and* the same content upscaled to 2240x2016 are both accepted — the regression that had silently rejected all production frames. `test_still_rejects_truly_dark_frame:68` asserts `reason == "brightness"`, `test_still_rejects_low_resolution_frame:75` asserts `reason == "resolution"`, and `test_thresholds_are_env_tunable:97` asserts the env overrides actually flip the verdict. Two tests pin the thresholds inside the measured dataset gap (`6.4 < BLUR_THRESHOLD < 88.8`; `BRIGHT_MIN <= 55.5` and `BRIGHT_MAX >= 67.7`).
- **RUNTIME EVIDENCE**: I ran `check_frame` over all 13 committed real frames. All 13 returned `ok=True` with real measurements, e.g. `open_circuit_2.jpg` (the softest real board) `lap_var=82.9, brightness=77.7` and `short_circuit_1.jpg` (sharpest) `lap_var=470.9, brightness=77.9` — a live spread of 82.9-504.1 against a threshold of 30, i.e. the gate has ~2.8x headroom on the softest genuine board and 14x on the sharpest. This gate is also why the zero-detection result in claim 1 is a *model* result, not a *rejected-frame* result: every frame passed.

---

## Built a BM25 SOP retrieval layer over manufacturing standard operating procedures with industry synonym expansion, returning clause-level document citations

- **STATUS**: VERIFIED
- **IMPLEMENTATION**: `SOPRetriever` (`src/agent/rag.py:13-128`) globs `data/sops/*.md`, tokenises with a custom industrial tokenizer that preserves hyphens and underscores (`tokenize`, `src/agent/rag.py:36-41`), and builds a `rank_bm25.BM25Okapi` index over the full document text (line 68). `search` (line 73) first expands the query through `SYNONYM_EXPANSIONS` (lines 19-27), a bilingual VI/EN defect vocabulary that maps e.g. `short_circuit` -> `["short circuit", "solder bridge", "han chap", "chap mach", "stencil", "kem han"]`, then scores with `get_scores` and returns `{sop_id, filename, score, snippet}`. A deterministic token-overlap fallback (lines 110-126) covers the case where BM25 returns nothing. The design choice of BM25 over vector search is recorded in `docs/adr/001-bm25-vs-vector-rag.md`.
- **FILE**: `src/agent/rag.py`, `data/sops/*.md` (6 documents), `docs/adr/001-bm25-vs-vector-rag.md`
- **TEST**: `tests/test_agent.py:99 test_sop_search_ipc610_and_nozzle_maintenance` asserts three separate queries each return the correct document at `top_k=1`: `"IPC Class 3 solder fillet"` -> `SOP-SMT-004`, `"reflow profile drift TAL thermocouple"` -> `SOP-SMT-005`, `"vacuum nozzle ultrasonic cleaning"` -> `SOP-SMT-006`. `tests/test_agent.py:24 test_sop_search_finds_solder_bridge` asserts `"short_circuit"` -> `SOP-SMT-001`.
- **RUNTIME EVIDENCE**: I instantiated `SOPRetriever` against `data/sops/` and ran five queries. Console output: `[RAG] Indexed 6 SOP manuals with Okapi BM25 engine.` and engine type `BM25Okapi`. Real results:
  ```
  "IPC-A-610 class 3 solder fillet criteria" -> SOP-SMT-004-ipc610-solder-criteria       score=5.79
  "reflow profile drift TAL thermocouple"    -> SOP-SMT-005-reflow-profiling-drift       score=5.08
  "vacuum nozzle ultrasonic cleaning"        -> SOP-SMT-006-pick-place-nozzle-maintenance score=2.80
  "solder bridge short circuit stencil"      -> SOP-SMT-001-solder-bridge                score=11.96
  ```
  The IPC-A-610 query returned the document titled "Tieu Chuan Danh Giia Mo Han Theo IPC-A-610 Class 3", and the reflow query returned "Kiem Soat & Hieu Chuan Lech Duong Cong Nhiet Lo Han Hoi Luu (Reflow Profile Drift)". Both are the right document.

  Honest limitation I found: the query `"line halt safety emergency"` does **not** retrieve the line-halt SOP. It returns `SOP-SMT-001-solder-bridge` at score 1.18 and `SOP-SMT-003-line-halt-safety` second at 1.16 — a near-tie that BM25 ranking gets wrong, because the safety SOP is dominated by Vietnamese terms the English query never hits. Retrieval was correct on 4 of my 5 probes, not 5 of 5.

---

## Designed a LangGraph multi-agent incident pipeline that proposes an intervention, blocks on a real interrupt() until a supervisor signs off, and is replay-safe on re-approval

- **STATUS**: VERIFIED
- **IMPLEMENTATION**: `QualityIncidentAgent._build_graph` (`src/agent/graph.py:56-75`) compiles a `StateGraph` over five nodes — `retrieve_sop`, `synthesize_rca`, `propose_action`, `await_supervisor_approval`, `execute_factory_action` — wired linearly `START -> ... -> END`, with a `MemorySaver` checkpointer for thread-scoped resume. `_node_await_supervisor_approval` (`src/agent/graph.py:77-94`) calls LangGraph's native `interrupt()` primitive, so the graph genuinely suspends mid-run rather than polling a flag. `run` (line 136) generates a `thread_id` and invokes the graph; `resume_approval` (line 173) resumes it with `Command(resume={approved, supervisor_id, plc_result, ticket_id})`. `_node_execute_factory_action` (`src/agent/graph.py:96-122`) is deliberately honest: it reports the *persisted* PLC outcome (`DISPATCHED`/`SIMULATED`/`EXECUTED` vs `FAILED`/`NOT_DISPATCHED`) rather than always claiming success, and it never actuates hardware itself — the service layer owns dispatch.
- **FILE**: `src/agent/graph.py`, `src/agent/subagents/sop_research_agent.py`, `src/agent/subagents/rca_analysis_agent.py`, `src/agent/subagents/intervention_agent.py`, `src/agent/checkpointer.py`, `tests/test_approval_safety.py`
- **TEST**: `tests/test_agent.py:76 test_agent_hitl_interrupt_and_resume` asserts the run returns `proposed_action == "HALT_LINE"`, `requires_hitl` is true, `"__interrupt__"` is present in the returned state, and that after `resume_approval(approved=True, supervisor_id="lead_qa_engineer")` the state carries `approval_status == "APPROVED"`, `approved_by == "lead_qa_engineer"`, and an `execution_result` containing `EXECUTED`. **The no-second-dispatch-on-replay proof does exist**, as two tests: `tests/test_approval_safety.py:280 test_replay_same_key_returns_stored_response_without_second_dispatch` calls `_resolve` twice with the same ticket and the same idempotency key against a `RecordingPLCBridge`, and asserts both calls return `EXECUTED` with an identical `ticket_id` while `plc.calls == [("halt_line", LINE_ID)]` — i.e. **exactly one dispatch**. `tests/test_approval_safety.py:182 test_repeated_approval_is_rejected_without_a_second_plc_dispatch` adds the persisted-column assertions: `ticket.status == "EXECUTED"`, `ticket.resolution_key == "approval-once"`, `ticket.plc_status == "SIMULATED"`, and a non-null `plc_result_json`.
- **RUNTIME EVIDENCE**: I ran the graph end to end. `run(defect_class="short_circuit", consecutive_count=3, thread_id="evidence-thread-01")` returned `proposed_action=HALT_LINE`, `requires_hitl=True`, `approval_status=PENDING`, `sop_citations=['SOP-SMT-001-solder-bridge', 'SOP-SMT-004-ipc610-solder-criteria']`, and `__interrupt__` was present with payload `{"action": "HALT_LINE", "line_id": "SMT-LINE-01", "defect_class": "short_circuit", "consecutive_count": 3, "rca_summary": "**Phan tich nguyen nhan goc (RCA)**: Phat hien 3 loi lien tiep..."}`. The graph was suspended, not completed. Then `resume_approval(approved=True, supervisor_id="lead_qa_engineer", plc_result={"status":"SIMULATED"})` returned `approval_status=APPROVED`, `approved_by=lead_qa_engineer`, `execution_result="EXECUTED: Action HALT_LINE dispatched to line SMT-LINE-01 (PLC=SIMULATED)."`. Both tests pass in my run.

  One thing to be careful about in an interview: the LLM path is **not** exercised. `src/runtime_flags.py:82-90` blanks `GROQ_API_KEY` under a test runner, so `RCAAnalysisAgent` takes its deterministic template branch. I never called Groq. The graph topology, the interrupt, the resume, and the replay-safety are verified; the LLM's contribution to the RCA text is `NOT VERIFIED`.

---

## Built a Modbus TCP client with fail-closed write validation, validated end-to-end against a virtual PLC server (not yet run against physical hardware)

- **STATUS**: PARTIALLY VERIFIED — the protocol path is fully verified; **no physical PLC was involved at any point**
- **IMPLEMENTATION**: `PLCBridge` (`src/industrial/plc_bridge.py:12-304`) speaks Modbus TCP over `pymodbus` `ModbusTcpClient`, with a documented register map: coils 0-5 (conveyor run, halt, rework divert, tower red/yellow/green — lines 19-25) and holding registers 0-3 (total inspected, defect count, yield x100, last defect code — lines 27-31). **Fail-closed validation is the core of the design.** `_parse_mode` (lines 73-80) rejects any `APEX_PLC_MODE` other than `simulation`/`hardware`. `connect` (lines 82-102) refuses outright unless `mode == "HARDWARE"`. Every command checks `is_connected` first and returns `_hardware_unavailable_result` with `status="FAILED"` (lines 146-153) rather than degrading to simulation. `_write_succeeded` (lines 132-137) treats a Modbus error response as failure, and `_write_coils` (lines 139-144) **aborts the remaining coil sequence the instant one is rejected**. `simulation_fallback` is still accepted by the constructor for backwards compatibility, but the comment at lines 44-46 is explicit that it "can never turn a failed hardware write into a simulated success". `VirtualModbusServer` (`src/industrial/simulator_plc.py:18-79`) is a pymodbus TCP server on a background event loop, so the client is exercised over a real socket.
- **FILE**: `src/industrial/plc_bridge.py`, `src/industrial/simulator_plc.py`, `docs/adr/002-modbus-tcp-virtual-plc-simulator.md`
- **TEST**: `tests/test_plc_bridge.py` (7 tests, all passing in my run). `test_plc_bridge_with_virtual_modbus_server:41` starts the virtual server and asserts a real connection plus `status == "DISPATCHED"` for halt, resume, and telemetry. `test_hardware_stops_remaining_coil_writes_after_rejected_response:121` is the sharpest one: it asserts `status == "FAILED"`, `error == "PLC_WRITE_REJECTED"`, and that `RejectingClient.calls == [(COIL_HALT_LINE, True)]` — proving the sequence stopped after the first coil rather than blindly pushing all five. `test_hardware_mode_rejects_unavailable_plc:80` asserts `FAILED` + `PLC_UNAVAILABLE`. `test_simulation_mode_never_connects_or_writes_to_hardware:91` patches `ModbusTcpClient` and asserts `client_class.assert_not_called()`. `test_hardware_write_exception_returns_failed_without_simulation_claim:101` and `test_telemetry_write_exception_returns_structured_failure:148` assert a raising write becomes `FAILED` carrying the exception text.
- **RUNTIME EVIDENCE**: I started `VirtualModbusServer` on 127.0.0.1:5077 and drove a real Modbus TCP session through it.
  - `PYMODBUS_AVAILABLE = True`; virtual server started: `True`. `bridge.mode = HARDWARE`, `is_connected = True`.
  - **Real coil write:** `halt_line()` -> `{"status": "DISPATCHED", "hardware_connected": true, "conveyor_running": false, "tower_light": "RED"}`, and an independent `read_plc_status()` read back `halt_triggered=true, conveyor_running=false, tower_light="RED"` **from the server**, not from local state. pymodbus logged the actual Modbus PDUs on the wire, e.g. `recv: ... 0x1 0x5 0x0 0x1 0xff` (FC05 write coil 1 = ON) and `recv: ... 0x1 0x5 0x0 0x0 0x0` (FC05 write coil 0 = OFF).
  - **Real holding-register write:** `update_telemetry(total=100, defects=5, yield_rate=95.0, defect_code=2)` -> `status=DISPATCHED`, wire `recv: ... 0x1 0x10 0x0 0x0 0x0 0x4 ...` (FC16 write multiple registers), and reading the registers straight off the server gave `[100, 5, 9500, 2]`.
  - **Rejected write returns FAILED:** against closed port 59999, `halt_line()` -> `{"mode": "HARDWARE", "status": "FAILED", "error": "PLC_UNAVAILABLE", "hardware_connected": false}`.
  - **Simulation never opens a socket:** with `ModbusTcpClient` patched, `client_class.called == False` and `halt_line()` returned `SIMULATED`.
  - **No physical PLC, no Siemens/Omron/Beckhoff device, and no production SMT line was connected, contacted, or commanded at any point.** `APEX_PLC_MODE` defaults to `simulation` (`src/industrial/plc_bridge.py:48`) and I never changed that default. Everything above happened over loopback against a Python process. That is a legitimate integration test and a fair CV line, but it is not hardware validation.

---

## Designed a relational schema with composite-key idempotency, per-ticket optimistic concurrency, and a transactional outbox

- **STATUS**: VERIFIED
- **IMPLEMENTATION**: Ten tables in `src/backend/models.py`: `production_lines`, `inspections`, `mes_tickets`, `audit_logs`, `idempotency_keys`, `jobs`, `outbox_events`, `dead_letters`, `audit_events`, `model_registry`. **`IdempotencyKey` uses a composite primary key `(caller_scope, idem_key)`** (`src/backend/models.py:71-72`) — note for accuracy this is a composite PK, not a separate unique index; `get_indexes` reports no additional unique index. `InspectionService.resolve_ticket` (`src/backend/service.py:176-376`) uses it as a per-ticket scope: `idem_scope = f"mes_resolve:{ticket_id}"` (line 198), so a caller-supplied key can never resolve a different ticket. Replay is served from the stored response at lines 203-210, *before* the ticket, agent, or PLC are touched. The **optimistic-concurrency reservation** is the conditional `UPDATE` at lines 214-225 — `WHERE ticket_id = ? AND status = 'PENDING_APPROVAL'` set to `RESOLVING` — and it gates on `reserved.rowcount != 1` (lines 226-228) to return `CONFLICT`. The idempotency row and the `OutboxEvent` are written in one `db.begin_nested()` block (lines 346-375), with the losing racer falling back to reading the winner's stored response on `IntegrityError`. `src/backend/database.py:200-239` adds cross-engine `ALTER TABLE` migrations for `thread_id`, `resolution_key`, `plc_status`, `plc_result_json` so old SQLite *and* PostgreSQL databases are upgraded in place.
- **FILE**: `src/backend/models.py`, `src/backend/database.py`, `src/backend/service.py`
- **TEST**: `tests/test_workflow_tables.py:15 test_shared_contract_tables_registered` asserts all six durable-ops tables are registered on `Base.metadata` (schema-shape only, not behaviour). The behavioural proofs are `tests/test_approval_safety.py:280` (idempotent replay, one dispatch) and `tests/test_approval_safety.py:141 test_duplicate_pending_ticket_is_not_created` (asserts exactly one `PENDING_APPROVAL` row survives a second cascade). `tests/test_backend.py:178 test_duplicate_resolution_returns_http_conflict` asserts the conflict surfaces as HTTP 409, and `tests/test_backend.py:47 test_ensure_thread_id_column_adds_when_missing` actually drops the column and re-runs the migration.
- **RUNTIME EVIDENCE**: I created a throwaway SQLite database, ran `init_db()`, and inspected the live schema. `inspect(engine).get_table_names()` returned all ten tables. `get_pk_constraint("idempotency_keys")` -> `constrained_columns == ['caller_scope', 'idem_key']`. Then, against real rows:
  - **Idempotent replay:** two `resolve_ticket` calls, same ticket, same key -> call#1 `EXECUTED` with `plc_calls=[('halt_line','SMT-LINE-01')]`, call#2 `EXECUTED`, `plc_calls` still exactly one entry, and the two response dicts compared **identical**. The stored row was `('mes_resolve:TICK-PROBE-3134A0', 'idem-91d42f', 'completed')`, and 1 `OutboxEvent` was written.
  - **Optimistic concurrency:** resolve#1 -> `EXECUTED`; resolve#2 with a *different* key -> **`CONFLICT` "Ticket has already been resolved or is being resolved."** with no second PLC call.
  - **Raw constraint:** inserting a duplicate `('c','k')` raised `IntegrityError`; inserting `('other','k')` succeeded, confirming the key is scoped per caller rather than globally unique.

---

## Built a runner-independent test-database guard that stops the suite from binding to production, and proved it in both directions

- **STATUS**: PARTIALLY VERIFIED — the guard itself is proven in both modes; one of the three shipped tests fails on Windows for an unrelated portability reason
- **IMPLEMENTATION**: The root cause is documented at `src/runtime_flags.py:8-14`: `tests/conftest.py` is a *pytest-only* hook, so `python -m unittest discover tests/` (documented as co-equal) bypassed it, `load_dotenv()` pulled the real Neon `DATABASE_URL` out of `.env`, and the destructive `setUp()` in `tests/test_backend.py` truncated production tables. The fix is to move the guard into production code: `is_test_process()` (`src/runtime_flags.py:35-68`) detects a test runner from markers a production process cannot produce (`PYTEST_CURRENT_TEST`, `pytest` in `sys.modules`, or a `__main__` ending in `unittest/__main__.py` / `unittest/main.py` / `pytest/__main__.py`), deliberately avoiding the unreliable `sys.argv[0]`. `src/backend/database.py:31-39` consumes it **before** `load_dotenv()` (line 42) and overwrites `DATABASE_URL` with `TEST_DATABASE_URL`; because `load_dotenv()` never overrides a pre-existing variable, the throwaway SQLite wins for the rest of the process. The same gate also blanks `GROQ_API_KEY` via `effective_groq_key` (`src/runtime_flags.py:82-90`), so unit tests cannot make live calls to `api.groq.com`. Both guards share one detection function so they cannot drift apart.
- **FILE**: `src/runtime_flags.py`, `src/backend/database.py:26-42`, `tests/conftest.py`, `tests/test_db_guard.py`, `tests/test_db_isolation.py`, `tests/test_llm_guard.py`
- **TEST**: `tests/test_db_guard.py` (3 tests). `test_unittest_runner_uses_throwaway_database:86` and `test_pytest_runner_uses_throwaway_database:91` each spawn a **subprocess** with `DATABASE_URL` pointed at an unreachable simulated production URL (`postgresql://probe:probe@127.0.0.1:1/probe_never_reached`, line 31), with `PYTEST_CURRENT_TEST` and `APEXINSPECT_TEST_MODE` explicitly removed from the child env (line 58) so the test proves the detection is *runner-based* rather than inherited from an outer pytest. The generated probe asserts the backend is sqlite, that the path contains `factory_test.db`, and that the env var no longer points at a remote. `test_guard_does_not_hijack_normal_processes:96` asserts the converse: a non-test script keeps its own `DATABASE_URL`. Corroborated by `tests/test_db_isolation.py:14 test_engine_is_local_sqlite_not_remote` and `tests/test_llm_guard.py:38 test_key_is_blanked_inside_a_test_runner`.
- **RUNTIME EVIDENCE**: I reproduced the guard by hand in both modes with the same hostile `DATABASE_URL`:
  ```
  ---- TEST (APEX_FORCE_TEST_MODE=1) ----
    [Database][TEST GUARD] Non-SQLite DATABASE_URL detected in a test run -> forcing throwaway SQLite to protect remote data: sqlite:///./factory_test.db
    PROBE_DIALECT=sqlite   PROBE_URL=sqlite:///./factory_test.db   PROBE_ENV=sqlite:///./factory_test.db

  ---- PROD (APEX_FORCE_TEST_MODE unset) ----
    PROBE_DIALECT=sqlite   PROBE_URL=sqlite:///./factory.db   PROBE_ENV=postgresql://probe:probe@127.0.0.1:1/prod_never_reached
  ```
  The guard fires in test mode and rewrites the URL; in production mode the environment variable is left completely untouched. Both verifications returned `True`.

  **Why one test fails.** In my run `tests/test_db_guard.py::test_guard_does_not_hijack_normal_processes` FAILED. The reason is Windows path handling, **not** a guard defect: SQLAlchemy percent-encodes backslashes in a Windows SQLite URL, so the child printed `POLICY_ENGINE=sqlite:///C%3A%5CUsers%5C...%5Cfactory_policy_check.db` while the test looked for the raw backslash form. The engine bound to exactly the file the test asked for, which is the behaviour the test is trying to prove. The test is correct on Linux (where CI runs it, green) and needs `sqlalchemy.engine.URL.render_as_string(hide_password=False)` instead of `str(engine.url)` to be portable.

---

## Maintained a 101-test suite split across unit, integration, and contract layers, and wired it into CI

- **STATUS**: VERIFIED (suite runs) / PARTIALLY VERIFIED (one platform-dependent failure, one test never executed)
- **IMPLEMENTATION**: 21 test files, 101 collected tests. There is no `pytest.mark.unit` / `integration` / `contract` taxonomy in the repo — the split below is mine, derived from what each test actually touches. **INTEGRATION (53)**: `test_plc_bridge.py` 7 (real TCP socket + pymodbus server), `test_approval_safety.py` 10 (SQLite + service + LangGraph + recording PLC double), `test_backend.py` 9 (SQLite + service + FastAPI handlers), `test_security_audit.py` 3 (SQLite + service + ASGI `TestClient` over real HTTP), `test_ingest.py` 6 (temp filesystem + in-memory SQLite), `test_prepare_dataset.py` 5, `test_stream.py` 2 (writes a real .mp4 and reads it through a worker thread), `test_db_guard.py` 3 and `test_llm_guard.py` 4 (both spawn subprocesses), `test_db_isolation.py` 2, `test_deploy_smoke.py` 1, `test_rca_groq_vcr.py` 1. **UNIT (33)**: `test_quality.py` 9, `test_vision.py` 7, `test_agent.py` 6, `test_sample_gallery.py` 5, `test_patching.py` 4, `test_health_contract.py` 2. **CONTRACT / static (15)**: `test_dashboard.py` 6 (all read `dashboard/*.html,css,js` or `render.yaml` as text), `test_training_yaml.py` 4 (1 parses `models/training_data.yaml`, 3 read `models/train.py` as a string), `test_workflow_tables.py` 1 (metadata introspection), `test_llm_guard.py` 2 (grep `src/` and the RCA module for forbidden patterns), `test_deploy_smoke.py::test_static_dashboard_contract_remote` 1, `test_vision.py::test_canonical_onnx_model_is_deployed` 1 (`os.path.getsize`).
- **FILE**: `tests/` (21 files), `tests/conftest.py`, `.github/workflows/ci.yml`
- **TEST**: the suite is its own evidence. Precisely: **86 of 101 tests execute production code and assert a runtime outcome** (a ticket status, a line state, a PLC dispatch count, a persisted DB row, a measured latency, a retrieved document, an engine binding); **15 of 101 are file-content or schema-shape assertions** that would still pass if the runtime behaviour regressed. Those 15 are worth keeping — they pin provenance and stop config rot — but they should not be counted as behavioural coverage. Of the 86, 84 passed in my run, 1 failed (the Windows portability issue above) and 1 was skipped.
- **RUNTIME EVIDENCE**: `.\.venv\Scripts\python.exe -m pytest tests/ -q --tb=short` -> **`1 failed, 99 passed, 1 skipped in 15.71s`** (a second run reproduced `1 failed, 99 passed, 1 skipped in 12.82s`).
  - The failure is `tests/test_db_guard.py::test_guard_does_not_hijack_normal_processes`, for the Windows URL-encoding reason given above.
  - The skip is `tests/test_rca_groq_vcr.py:54`: "cassette missing and no real GROQ_API_KEY; record once with APEX_ALLOW_LIVE_LLM=1 and a real key to create it". `tests/cassettes/` contains only `.gitkeep`, so **the VCR replay path has never actually been executed** — the only real-Groq test in the repo is inert. Treat the LLM integration as `NOT VERIFIED`.
  - Caveat on the headline latency test: `tests/test_vision.py:56 test_detector_inference_latency_under_threshold` asserts `len(detections) >= 1`, but line 65 sets `ALLOW_GT_FALLBACK=true`, so on a frame where the network finds nothing the assertion is satisfied by **injected ground truth**, not by a model prediction. The test proves latency, not detection.
  - Suite runtime is ~13-16s, versus the 49.6s the LLM guard fixed (`src/runtime_flags.py:17-18`).

---

## Configured GitHub Actions CI (test + Docker build) and CD (GHCR image + Render deploy trigger)

- **STATUS**: VERIFIED
- **IMPLEMENTATION**: `.github/workflows/ci.yml` runs on push/PR to main and master with `contents: read`. The `test` job (line 17) checks out, sets up Python 3.11, runs `pip install -r requirements.txt` (the **full** file, including ultralytics and torch), installs `pytest`, prints the resolved versions, then runs `python -m pytest tests/ -q --tb=short` (line 39), followed by a second explicit safety-gate invocation over `test_plc_bridge.py test_approval_safety.py test_agent.py test_backend.py test_llm_guard.py` (line 45). The `docker-build` job (line 47) runs `docker/build-push-action@v5` with `push: false` — a build smoke test that validates the Dockerfile without publishing. `.github/workflows/cd.yml` builds and pushes to GHCR on `main` (with `packages: write` and gha cache), then `deploy-render-api` fires the Render deploy hook, or the Render API if the hook secret is absent, and exits 0 with a skip message if neither is configured.
- **FILE**: `.github/workflows/ci.yml`, `.github/workflows/cd.yml`, `.github/workflows/deploy-azure.yml`, `Dockerfile`, `render.yaml`
- **TEST**: the CI configuration is the test. There is no test of the workflows themselves.
- **RUNTIME EVIDENCE**: `gh run list --repo imtarget05/ApexInspect-AI --limit 10` returned **10 runs, all `completed success`, no failures**:
  ```
  completed  success  chore: trigger CI verification                          CI    35360208078  2m27s  2026-09-18T15:04:52Z
  completed  success  chore: trigger CI verification                          CD    35360208082  49s    2026-09-18T15:04:52Z
  completed  success  fix(tests): isolate suite from prod DB and live LLM      CI    35248639639  2m31s  2026-09-17T16:46:20Z
  completed  success  fix(tests): isolate suite from prod DB and live LLM      CD    35248639694  50s    2026-09-17T16:46:20Z
  completed  success  fix(tests): isolate suite from prod DB and live LLM      Azure  35248639613  2m9s   2026-09-17T16:46:20Z
  completed  success  test(security): make X-API-KEY assertions env-indep.    CI    35232855356  2m21s  2026-09-17T14:20:04Z
  completed  success  test(security): make X-API-KEY assertions env-indep.    CD    35232855316  42s    2026-09-17T14:20:04Z
  completed  success  docs(deploy): promote Render Static Site to SoT        CI    35230840484  2m45s  2026-09-17T14:01:33Z
  completed  success  docs(deploy): promote Render Static Site to SoT        CD    35230840424  1m12s  2026-09-17T14:01:33Z
  completed  success  feat(ui): replace Streamlit with static dashboard      CD    35229827535  1m3s   2026-09-17T13:52:10Z
  ```
  (The 5th row's workflow is "Deploy to Azure Container Apps", from `deploy-azure.yml`.) Note the newest CI run is dated **2026-09-18**, which is before this audit; the green history is real but is not a guarantee about the current tree.
  I also reproduced the `docker-build` job locally. `docker build --no-cache -t apexinspect-audit:evidence .` completed successfully and produced a **977 MB** image. `docker inspect` confirms the image is hardened as the Dockerfile intends: `USER=apexinspect` (non-root), `EXPOSED={"7860/tcp":{}}`, and the branching `HEALTHCHECK` is present verbatim. I then ran it and exercised the real HTTP surface:
  ```
  docker run -d --name apex-evidence -e APP_MODE=api -e PORT=7860 -p 17860:7860 apexinspect-audit:evidence
  GET  /health/live      -> HTTP 200 {"status":"ok","service":"apexinspect-gateway","version":"1.0.0"}
  GET  /health/ready     -> HTTP 200 {"status":"ready","checks":{"config":"ok","database":"ok","model_bundle":"ok"}}
  ```
  So the container, the FastAPI gateway, the SQLite fallback, and the non-mutating readiness probes all work as advertised.
  Side observation: `.dockerignore` excludes `.env`, `.git`, `docs`, `tests`, `scripts` and `*.db` but **not `.venv`**, so my first local attempt shipped a 1.38 GB build context containing the virtualenv before the image layer was cached. CI never sees this because a fresh checkout has no `.venv`, so the CI job is not slowed by it - but a first-time local build is, and it is a one-line `.dockerignore` fix.

---

## Claims that are overstated in the repository and must NOT go on a CV as written

These three were called out for review. I verified each; the honest version follows.

### "Immutable audit log" — NOT TRUE. Do not claim immutability.

`AuditLog` is declared at `src/backend/models.py:53-64` with the docstring *"Immutable audit trail for industrial safety decisions and line control actions"*, and the `InspectionService` class docstring repeats it at `src/backend/service.py:27`. **There is no append-only constraint, no trigger, and no hash chain.** `src/backend/service.py` writes ordinary rows via `db.add(audit); db.commit()` (lines 114-123, 159-168, 302-315, 398-407), and nothing anywhere prevents `UPDATE` or `DELETE`. The test at `tests/test_security_audit.py:32` only asserts a row *exists* with the right `action` and `operator_id` — it never asserts the row cannot be changed.

I proved the mutability rather than inferring it. Against a live SQLite database holding real audit rows:
```
audit_logs row count before: 2
target row: AUD-2C89C40B APPROVE_ACTION
UPDATE succeeded -> "TAMPERED by auditor"
DELETE succeeded -> count now: 1
CHECK constraints on audit_logs: []
sqlite_master triggers for audit_logs: 0
```

**What the feature actually provides:** durable, queryable, attributable records of who did what to which line and ticket, with `operator_id`, `action`, `line_id`, `ticket_id`, `source_ip` and a UTC timestamp, written in the same transaction as the state change they describe. That is a real and useful audit trail. It is **append-in-convention, not append-only in-enforcement**. Say exactly that. A genuine fix would be a `BEFORE UPDATE OR DELETE` trigger that raises, or a `prev_hash`/`row_hash` chain — neither exists.

### "PLC control" — overstated. No hardware has ever been touched.

`APEX_PLC_MODE` defaults to `"simulation"` at `src/industrial/plc_bridge.py:48`, and `connect()` returns `False` immediately unless the mode is `HARDWARE` (lines 84-87). `test_simulation_mode_never_connects_or_writes_to_hardware:91` proves the default path never constructs a `ModbusTcpClient`. `src/backend/service.py:246-251` explicitly constructs `PLCBridge(mode="simulation")` on the agent-driven path, and the module docstring at `tests/test_approval_safety.py:12` states the intent: "All tests use a recording PLC double / loopback simulator only - no test may emit a command to real hardware."

**Rewrite the claim as:** *"Modbus TCP client for SMT line actuators (conveyor interlock, line halt, pneumatic reject diverter, andon tower light) with fail-closed write validation, verified end-to-end over a real TCP socket against a virtual Modbus server in integration tests; not yet run against physical hardware."* The register map, the coil sequencing, the error-frame handling, and the fail-closed semantics are all genuinely implemented and genuinely tested. The only thing missing is a real PLC on the other end.

### The fail-open vision path — verified, and it belongs in a security review, not on a CV.

`src/vision/detector.py:272-274`: when the model fails to load, `infer()` takes the `else` branch, calls `time.sleep(0.025)`, and returns `detections = []` (unless `ALLOW_GT_FALLBACK` is on). `render_annotations` at line 288 then paints **`STATUS: PASS (NO DEFECTS)`** onto the frame, in green, because it keys purely off `len(detections) == 0`. A detector that is completely non-functional is therefore reported to the operator as a board that passed inspection. In a quality-gate context that is the dangerous direction: defects ship.

I confirmed it by constructing a detector with `use_onnx = False` and calling `infer()` on a real frame:
```
use_onnx=False                            -> detections=[]  latency_ms=25.20   PASS painted on frame: yes
use_onnx=False + ALLOW_GT_FALLBACK=true   -> detections=[{'class':'short_circuit','bbox':[10,10,50,50],'confidence':0.95}]  latency_ms=25.52
```

Note the 25 ms is a `time.sleep(0.025)`, not measured compute. Two distinct problems, then: a missing model is indistinguishable from a clean board at the presentation layer, and a missing model is indistinguishable from a fast model in the telemetry.

There is a partial mitigation worth knowing: `tests/test_vision.py:113 test_infer_does_not_fake_gt_in_production` asserts that with `ALLOW_GT_FALLBACK=false` the detector does **not** return ground truth as predictions. That test passes. It does not touch the PASS-labelling problem, and no test covers it.

### Additional overstatement found during this audit

`models/MODEL_CARD.md:22` claims a "<= 35 ms (~28 FPS, NFR >= 25 FPS)" latency target, and the committed `benchmark_results.json` reports 43.36 ms / 23.1 FPS — already above that target. My measurement on different hardware was 86.2 ms / 11.6 FPS, and `scripts/check_benchmark_target.py` would exit 1. The 35 ms figure should not appear on a CV as a met target.

### Not verified, and worth knowing before an interview

- **The live LLM path.** `tests/test_rca_groq_vcr.py` is skipped because `tests/cassettes/` holds only `.gitkeep`, so the one test that would exercise a real `CloudGroqProvider` HTTP call has never run. Everything agent-related I verified used the deterministic offline RCA fallback.
- **`models/train.py` and the Colab notebook** were not executed — no GPU, and retraining is out of scope. The provenance in `models/MODEL_CARD.md` is plausible and internally consistent with `models/training_data.yaml`, but I did not reproduce a training run.
- **Docker runtime beyond the API mode.** The image builds and `APP_MODE=api` serves and self-reports healthy, but I did not exercise the `APP_MODE` Streamlit branch or a GHCR push, so the CD path is evidenced only by the recorded run history.
- **PostgreSQL paths.** The guard, the migrations, and `with_for_update()` all have PostgreSQL-specific branches, but I only exercised SQLite. The Neon-hosted configuration is `NOT VERIFIED`.
- **Physical PLC / camera hardware.** Not involved anywhere, by design and by configuration.

---

## Known Security Issues (fix before showcasing)

- **Hardcoded default API key, guarding only one of the write routes.** `src/backend/main.py:96` falls back to a hardcoded literal when `$API_KEY` is unset (`Potential committed secret detected: src/backend/main.py:96`), so a deployment that forgets the env var runs with a publicly known, source-visible credential. Worse, `verify_api_key` is attached only to `POST /api/v1/mes/action` (`src/backend/main.py:142`); `POST /api/v1/inspections` (`src/backend/main.py:123`) has no `dependencies=[...]` and is fully unauthenticated while still writing to `inspections` and opening MES tickets. **Confirmed live against the built container**, with no `API_KEY` set and no header sent: `POST /api/v1/inspections` returned `HTTP 200 {"inspection_id":"INSP-49B2DCD3","status":"RECORDED",...}` — an anonymous write succeeded — while `POST /api/v1/mes/action` on the same container returned `HTTP 401`. `tests/test_security_audit.py:102 test_api_key_authentication_enforced` only exercises the *guarded* route, so nothing in the suite would catch this.
- **CORS wildcard combined with credentials.** `src/backend/main.py:31-32` sets `allow_origins=["*"]` together with `allow_credentials=True`. Any origin can make credentialed cross-site requests to the gateway, and the two settings are mutually contradictory in the spec, so browsers and frameworks resolve it inconsistently.
- **`ALLOW_GT_FALLBACK` returns ground truth as predictions.** `src/vision/detector.py:259` reads the flag; lines 270-271 substitute the supplied `ground_truth_defects` for the model's output whenever inference returns nothing, and line 274 does the same in the no-model branch. Any environment that sets this to `true` renders labelled ground-truth boxes as if they were network predictions, silently inflating every accuracy number. It defaults to `false` and `tests/test_vision.py:113` guards that default, but nothing prevents a deploy from flipping it.
- **The detector fails open to a green PASS.** Covered in full above: `src/vision/detector.py:272-274` plus the label at line 288. A model that fails to load is reported as `STATUS: PASS (NO DEFECTS)` with a synthetic 25 ms latency.
- **Credential-shaped literals are committed in two places.** `tests/conftest.py:29` and `.env.example` both hardcode a Groq-style placeholder token. It is a placeholder rather than a live secret, and in `conftest.py` it doubles as a useful tripwire (`RCAAnalysisAgent` treats that exact value as "not configured"), but both will trip secret scanners in any CI log or diff.

**Secret scan:** no live credential was found committed in the tree, and no secret value is reproduced anywhere in this document. `Potential committed secret detected: src/backend/main.py:96` (a default API key is hardcoded as a fallback when `$API_KEY` is unset) and `Potential committed secret detected: tests/conftest.py:29` (a Groq-style placeholder token, alongside the same placeholder in `.env.example`). Both are placeholders or defaults rather than working credentials, and no real `.env` is present in the working tree or tracked by git.
