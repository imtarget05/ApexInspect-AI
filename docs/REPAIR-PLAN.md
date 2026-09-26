# REPAIR PLAN — ApexInspect-AI

Audit: 2026-09-25. Static deep audit (source, config, tests, CI, Docker, model card were all read).

> **Execution status: NOT VERIFIED.** The dependencies (`ultralytics`, `onnxruntime`, `kagglehub`, `opencv`) pull multiple GB and were deliberately not installed. Nothing in this document claims a test was run. Every status below is `IMPLEMENTED` / `PARTIALLY IMPLEMENTED` / `NOT VERIFIED` based on reading the code, except where a test file proves the behaviour by asserting it.

---

## Current State

Factory PCB defect inspection platform: YOLOv8n exported to ONNX for edge inference, a LangGraph agent with human-in-the-loop approval that diagnoses root cause from a BM25 SOP corpus and proposes factory actions, a Modbus TCP bridge for PLC actuation, FastAPI + Streamlit, SQLAlchemy over Neon Postgres with a SQLite fallback, and a deterministic test-isolation guard.

102 test functions across 24 files. Three GitHub Actions workflows.

Two of the three headline claims are overstated, and the vision layer **fails open in a way that would be a safety incident on a real line**.

---

## Verified Working Features

Read in source; where a test proves it, the file is cited.

- **Real trained + exported ONNX detector, committed with provenance.** `models/yolov8n_pcb_defect.onnx` (~12 MB), `models/MODEL_CARD.md:1-27` (Colab T4, `akhatova/pcb-defects` dataset, 6 classes, IO shapes). `.gitignore:23-24` has an explicit `!models/yolov8n_pcb_defect.onnx` exception — committing this binary is the **right** decision, and the model card is unusually good practice. Binary contents: `NOT VERIFIED` (cannot inspect a binary by reading it).
- **Real ONNX inference path** — `src/vision/detector.py:247` `infer` → quality gate `src/vision/quality.py check_frame` (`:262`) → CLAHE + letterbox (`detector.py:155-167`) → `session.run` (`:268`) → YOLOv8 decode + NMS (`_postprocess_yolov8:169-245`). Covered by `tests/test_vision.py:39-113`.
- **An honest test that prevents a specific cheat** — `tests/test_vision.py:113` `test_infer_does_not_fake_gt_in_production`. Whoever wrote this knew about the `ALLOW_GT_FALLBACK` trap and wrote a test against it.
- **Real BM25Okapi SOP retrieval** — `src/agent/rag.py:43-71` over six real SOP files in `data/sops/` including IPC-A-610 Class 3 (`SOP-SMT-004-ipc610-solder-criteria.md`), reflow drift (`-005`), and nozzle maintenance (`-006`). Synonym expansion at `:19-27`; token-overlap fallback at `:110-126`. ADR-001 honestly states there is no vector layer.
- **Real LangGraph HITL** — `interrupt()` at `src/agent/graph.py:80`, resume via `Command(resume=…)` at `:191-194`, a durable `IncidentCheckpointer` (`:53,203-209`) alongside `MemorySaver` (`:52`).
- **Real Modbus TCP client with fail-closed write validation** — `src/industrial/plc_bridge.py:82-102` `ModbusTcpClient`, coils/registers at `:19-31`, write-ack validation at `:132-143`, FAILED on rejected write at `:159-173`. Plus a `VirtualModbusServer` in `src/industrial/simulator_plc.py`. Covered by `tests/test_plc_bridge.py` (7 tests, including "no simulated success" and "telemetry failure").
- **Real trigger engine** — 3-consecutive-defect and yield-drift >15% over a 10-row window (`src/backend/service.py:79-87`), with duplicate-pending suppression using `with_for_update()` on Postgres (`:62-74`).
- **Real relational schema with an idempotency guard** — `src/backend/models.py`; composite unique index on `IdempotencyKey(caller_scope, idem_key)`; an optimistic-concurrency conditional UPDATE `PENDING_APPROVAL → RESOLVING` as the reservation (`service.py:214-229`); an outbox (`service.py:346-376`). `tests/test_approval_safety.py:171-292` verifies exact dispatch counts and **replay-safety (no second dispatch on replay)**.
- **A genuinely excellent test-isolation story, backed by regression tests.** `src/runtime_flags.py:35-68` `is_test_process` is consumed *before* `load_dotenv()` at `src/backend/database.py:31-42`, forcing `sqlite:///./factory_test.db`. `tests/test_db_guard.py:40-96` runs **both runners in a subprocess** to prove it. `effective_groq_key:82-90` blanks the LLM key under test runners, and `tests/test_llm_guard.py:112` is a meta-test asserting no ungated key read exists. This is the strongest engineering narrative in the repository and it is fully backed by code.
- **Postgres → SQLite fallback** — `src/backend/database.py:60-100`, with connect-timeout and verify-ping before accepting (`:80-92`).
- **CI with no bypasses** — `.github/workflows/ci.yml` runs pytest on 3.11 plus a second "Safety gates" pytest run over PLC honesty / idempotent actuation / LLM guard (`:44-45`), then a Docker build smoke test (`:47-63`). `continue-on-error` in CI: **0**.

---

## Broken Features

### B1 — The vision layer fails open and reports every board as PASS · **P0, safety-relevant**

`src/vision/detector.py:109-114`: if the ONNX file is missing, or `onnxruntime` fails to initialise, the detector prints "Running in simulation mode", sets `use_onnx=False`, and `infer` (`:272-274`) does `time.sleep(0.025)` and **returns `[]`** — an empty detection list.

`render_annotations:288` then renders that as `STATUS: PASS (NO DEFECTS)`.

So: a broken or missing model produces a stream of confident PASS verdicts with a **fabricated 25 ms latency**, and nothing in the response says the model is absent. On a real SMT line that is a quality-escape defect, not a demo inconvenience.

It is *partially* mitigated — `/health/ready` probes the model bundle (`src/backend/main.py:58-63,71-86`) — but the Streamlit path bypasses that probe, and a per-frame verdict is still wrong.

### B2 — A hardcoded default credential guards an actuation endpoint · **P0**

`src/backend/main.py:96` defaults the expected API key to the literal string `"dev-factory-key-secret"` when `API_KEY` is unset. The comparison at `:99` is a plain `==`, not `hmac.compare_digest`.

`render.yaml:27-28` marks `API_KEY` as `sync: false`, so **a misconfigured deployment runs on a publicly-readable default key** protecting the endpoint that halts a production line.

### B3 — The same Azure no-op gate as CreditFlow

`.github/workflows/deploy-azure.yml:41,48` gate on `if: env.AZURE_CREDENTIALS != ''` where the variable is only step-scoped (`:43,50`) → the job is green and deploys nothing.

---

## Half-implemented Features

### H1 — "Immutable AuditLog" is not immutable · **P1**

`src/backend/service.py:302-315` and `:114-123` write ordinary SQLAlchemy rows. There is **no append-only constraint, no hash chain, no trigger, no immutability enforcement**. The database user can UPDATE or DELETE audit rows at will. The README calls them "immutable". CreditFlow at least recomputes a SHA-256 over its ledger rows; this repository does not even do that.

### H2 — The PLC claim is simulation-only · **P1**

`src/industrial/plc_bridge.py:48` defaults `APEX_PLC_MODE` to `"simulation"`, and `src/backend/main.py:116` constructs `PLCBridge()` **with no mode argument**. Therefore every dispatch returns the string `"SIMULATED"` and never touches hardware.

To be fair to the codebase: the design is honest — `plc_bridge.py:159-173` returns FAILED on a rejected write, and the status string is never conflated with `DISPATCHED`. But the README's "control conveyor interlocks, pneumatic reject diverters, andon tower lights" describes capability that **has never been exercised against a physical PLC**.

### H3 — `ALLOW_GT_FALLBACK` returns ground truth as predictions

`src/vision/detector.py:259,270-271`: when `ALLOW_GT_FALLBACK=true` the detector returns ground-truth labels as its own predictions and swallows misses as passes. Correctly defaulted to `false` and documented as demo-only in `.env.example:27-29` and `models/MODEL_CARD.md:27`. Classify as `DANGEROUS_PROD_MOCK if ever set in production` — do not remove it, it is useful for demos, but keep the guard.

### H4 — Dead code after a `return`

`src/vision/detector.py:132`: `self._last_letterbox_info = None` sits **after** the `return` at `:130`, so it is unreachable.

### H5 — Neon Postgres fallback is silent

`src/backend/database.py:97-100` falls back to `sqlite:///./factory.db` with only a `print`. A production Postgres outage silently downgrades a factory deployment to a local SQLite file with **no health signal**.

---

## Documentation Claims Not Verified

| Claim | Status |
|---|---|
| "43.36 ms average latency (23.1 FPS)" | **NOT VERIFIED as a production number.** The harness is reproducible code (`scripts/benchmark_inference.py:25-62`, 5 warmup + 20 measured), but `benchmark_results.json:3` leaks an absolute local path and `:25` states the frames are **synthetic PCB frames**. It is a local simulated-frame measurement. The README itself concedes this at `:89`. |
| "≤35 ms on cloud 2-vCPU, not yet verified as achieved" | **Honest.** Explicitly disclaimed at `README:27,108,139`. `scripts/check_benchmark_target.py` exists. No CI latency gate — correctly disclosed as claim 15. Keep this wording; it is exactly the right way to do it. |
| "Immutable `AuditLog` table" | **FALSE as stated.** See H1. |
| "control conveyor interlocks… andon tower lights" | **NOT VERIFIED against hardware.** See H2. |
| "87 passed, 1 warning, exit code 0, repeated twice" | **NOT VERIFIED.** No CI log artifact, no badge JSON, no stored run output. 102 test functions exist, and `tests/test_rca_groq_vcr.py:45-50` **skips** when the cassette is absent — and `tests/cassettes/` contains only `.gitkeep`, so at least one of the "87 passed" is a skip, not a pass. |
| `X-API-KEY` protects the service | **PARTIALLY IMPLEMENTED.** It guards exactly **one** route (`main.py:142`). See S2. |
| "Neon Serverless PostgreSQL with automatic fallback to SQLite" | **IMPLEMENTED** — `database.py:60-100`. Code is real; no evidence of a live Neon connection. `CONFIGURED`. |
| "The Render deploy job skips gracefully when no hook is configured" | **IMPLEMENTED and honestly disclosed** — `cd.yml:70-80`, `README:222-223`. Note the corollary: a green CD can mean "nothing deployed". |
| "Rescued defect — production DB wipe; fixed by runner-independent guard" | **IMPLEMENTED and tested.** `src/runtime_flags.py:35-68` + `tests/test_db_guard.py:40-96`. This claim is fully true. |

---

## Security Problems

### S1 · P0 — Partial auth on an actuation-capable service

`X-API-KEY` (`main.py:92-109`) is applied to **exactly one** route: `POST /api/v1/mes/action` (`:142`).

Unauthenticated:
- `POST /api/v1/inspections` (`main.py:123`) — **telemetry write**, which feeds the 3-consecutive-defect trigger at `service.py:89`. An anonymous caller can inject three "defective" frames and manufacture a CRITICAL MES ticket and a `HALT_LINE` proposal.
- `GET /api/v1/lines/{id}/metrics` (`main.py:170`)
- `GET /api/v1/tickets/recent` (`main.py:193`)

There is no RBAC beyond the key: no roles, no per-line scoping, and `approved_by` is taken from the request body (`main.py:155` → `service.py:180`), so supervisor identity is self-asserted.

### S2 · P0 — CORS wildcard with credentials

`allow_origins=["*"]` **and** `allow_credentials=True` at `src/backend/main.py:29-35`. This is the classic misconfiguration: a browser will reject the response, but the intent expressed is to allow any origin with credentials.

### S3 · P1 — `API_KEY` default is a public secret

`Potential committed secret detected: src/backend/main.py:96` — key name: `API_KEY` default value. **Action: remove the default and fail closed in production.**

### S4 · Verifiable positives — keep these

- **No SQL injection.** All SQLAlchemy Core/ORM with bound parameters; the only string-built SQL is internal DDL with hardcoded column names (`database.py:214,236`).
- **No committed secrets.** `.env` is gitignored (`.gitignore:12`); `render.yaml:23-28` all `sync: false`.
- **API-key route is genuinely protected** — `tests/test_security_audit.py:102` asserts 401 without the key.
- **The test-isolation guard is a real security control**, not just hygiene: it is what prevents a test run from wiping the production database, and it is regression-tested in a subprocess.

---

## Testing Gaps

- **102 test functions** across 24 files. The genuinely behavioural ones are strong: `test_approval_safety.py` (10 tests — exact dispatch counts, replay-safety, no-actuation-on-reject, state transitions), `test_plc_bridge.py` (7), `test_backend.py` (9 — trigger conditions, 409 conflict, 502 on hardware failure), `test_security_audit.py`, `test_stream.py` (threaded camera).
- **Roughly 20 of 102 are file/config shape assertions**, not behaviour: `tests/test_dashboard.py:17-64`, `tests/test_deploy_smoke.py:23-29`, `tests/test_training_yaml.py:16-41`. Do not quote 102 as behavioural coverage.
- **`tests/test_rca_groq_vcr.py` permanently skips** — `tests/cassettes/` holds only `.gitkeep`. Either commit a cassette or drop the test.
- **No test asserts that a missing model is refused** — the exact gap that lets B1 exist. This is the most important missing test in the repository.
- **No test for hardware mode.** Nothing exercises `APEX_PLC_MODE=hardware` against a real or emulated broker, which is why H2 has gone unnoticed.
- **No load test**, and no latency gate in CI (correctly disclosed).
- `ci.yml:36` greps for a `chromadb` dependency that is not in the stack — cosmetic drift.

---

## Deployment Gaps

| Target | Classification |
|---|---|
| CI (pytest + safety gates + Docker smoke) | `CI_IMPLEMENTED`, and no bypass affordances |
| GHCR push | `CD_IMPLEMENTED` |
| Render API | `CD_IMPLEMENTED` but **skippable** — `exit 0` when no hook/key. Disclosed. |
| Render static dashboard | `CONFIGURED`, contract-tested locally |
| Neon Postgres | `CONFIGURED`, no cloud verification |
| Groq | `CONFIGURED_OPTIONAL` — deliberately off the critical path |
| Azure Container Apps | **BROKEN GATE** (B3) |
| **CLOUD_VERIFIED** | **NO** for anything. No CI step smoke-tests the live API. |

Docker: single `Dockerfile`, no compose. Coherent — `python:3.11-slim` + OpenCV system libs (`:6-11`), non-root `apexinspect` user (`:14,28`), `requirements-runtime.txt` excluding training deps (`:17-18`), `APP_MODE` switching API vs Streamlit on one port (`:44-47`), `EXPOSE 7860` matching Render. Caveat: `:21` `COPY . .` copies the whole repo; verify `.dockerignore` excludes `.venv` and the dataset directories.

---

## Recruiter-facing Problems

1. **"Immutable AuditLog" is not immutable.** A security-adjacent claim that is simply false. In a project whose pitch is manufacturing traceability, this is the finding a reviewer will check first.
2. **The PLC claim has never touched hardware.** "Directly control conveyor interlocks" reads as deployed capability. It is a well-built simulation.
3. **A missing model produces confident PASS verdicts.** Even though it is unlikely to be discovered without reading `detector.py`, if it *is* discovered it is the worst possible finding for a quality-control system.
4. **A hardcoded default key on an actuation endpoint, with an unauthenticated telemetry write that can trigger it.** Three findings that chain together into one attack path.
5. **"87 passed" is unverifiable and at least one of those is a skip.**
6. **No `RECRUITER-EVIDENCE.md`,** despite having one of the best test suites and a genuinely excellent incident story (the production-DB-wipe rescue) that would be a strong interview narrative if it were written down.

---

## Repair Tasks

### P0 — blocking

- [ ] **P0-1 Make a model-load failure fatal.** In `src/vision/detector.py:_initialize_engine` (`:109-114`), raise instead of falling back to zero detections. If a degraded read-only mode is genuinely needed for demos, gate it behind an explicit `APEX_ALLOW_SIMULATION=true` and stamp every response with `model_status: "degraded"` so no consumer can mistake it for a real verdict. **Add the missing test: a missing/corrupt ONNX file must not produce a PASS verdict.**
- [ ] **P0-2 Remove the `"dev-factory-key-secret"` default** at `src/backend/main.py:96`. Fail closed when `API_KEY` is unset in production, and compare with `hmac.compare_digest` at `:99`.
- [ ] **P0-3 Protect `POST /api/v1/inspections`** (`main.py:123`) with the same key, or add explicit rate limiting plus a signed-ingest credential. An anonymous write that can trigger `HALT_LINE` is not acceptable.
- [ ] **P0-4 Fix CORS** at `main.py:29-35` to an explicit allowlist with `allow_credentials=False`, or align it with a configurable origin list.
- [ ] **P0-5 Stop trusting `approved_by` from the request body** (`main.py:155` → `service.py:180`). Bind supervisor identity to the authenticated caller.

### P1 — important

- [ ] **P1-1 Make `AuditLog` actually append-only,** or reword the README. Preferred: a database trigger or a hash chain (`prev_hash`, `hash`) with a verification query — the same technique CreditFlow already uses in `pipeline/storage/ledger.py:516-543`, which can be cited as prior art within the portfolio.
- [ ] **P1-2 Prove one hardware-mode run,** or reword the PLC claim to "Modbus TCP client with a virtual server, validated by `tests/test_plc_bridge.py`; not yet run against physical hardware." The second option costs ten minutes and removes the overstatement entirely.
- [ ] **P1-3 Fix the Azure gate** in `deploy-azure.yml:41,48`, exactly as in CreditFlow.
- [ ] **P1-4 Make the Postgres→SQLite fallback non-silent** at `database.py:97-100` — set a `db_backend` field on `/health/ready` so a downgraded deployment is visible.
- [ ] **P1-5 Commit a VCR cassette** for `tests/test_rca_groq_vcr.py`, or delete the test so the pass count is honest.
- [ ] **P1-6 Reconcile the "87 passed" claim** with reality, and re-measure the benchmark with a stated hardware profile, or restate it as a local synthetic-frame measurement.
- [ ] **P1-7 Add a `APEX_PLC_MODE=hardware` test** against an emulated Modbus broker, so the hardware branch has any coverage at all.

### P2 — nice-to-have

- [ ] **P2-1** Move `detector.py:132` above the `return` at `:130`, or delete it.
- [ ] **P2-2** Add a `make bench` target and commit its output, so the latency number is regenerable.
- [ ] **P2-3** Add a `docker-compose.yml` (API + Postgres) so the project is demonstrable as a container.
- [ ] **P2-4** Remove the `.agents/skills/*` and `docs/superpowers/specs/*` vendored noise from the tree.
- [ ] **P2-5** Remove the absolute developer path from `benchmark_results.json:3`.
- [ ] **P2-6** Write `docs/RECRUITER-EVIDENCE.md` — the DB-wipe rescue story and the `test_db_guard.py` subprocess test make an excellent interview narrative that currently exists only as a README bullet.
