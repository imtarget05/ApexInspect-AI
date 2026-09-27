# ApexInspect-AI — Interview Evidence Kit

> Edge vision + SOP RAG + HITL PLC actuation.
> Measured: ONNX **43.36 ms avg (23.1 FPS)**, input `[1,3,640,640]`, 20 runs +
> 5 warmup (`benchmark_results.json`). Existing ADRs 001–003 (BM25 RAG, Modbus
> simulator, ONNX weights) + new ADR-004 (fail-closed detector + Groq RCA).
> Phase-2 addition: 6 detector/safety tests.

---

## 1. STAR story

**Situation.** SMT/PCBA lines need high-speed defect inspection tied to physical
actuation (conveyor e-stop, diverter, andon lights) — but dev has no hardware
PLC, cloud inference is too slow/expensive per frame, and auto-actuation
without human sign-off is a safety incident waiting to happen.

**Task.** Build edge inference + grounded RCA + HITL-governed PLC dispatch that
runs offline by default and never converts a failed hardware write into a
simulated success.

**Action.**
- YOLOv8n (Colab T4 trained) → ONNX Runtime CPU + SAHI high-res patching with
  NMS merge; letterbox pad=114; `PCBCameraSimulator` for offline frames.
- Okapi BM25 SOP RAG (ADR-001) with domain synonym expansion over 6 SOPs
  (IPC-A-610, reflow TAL/PWI, nozzle maintenance) feeding the LangGraph RCA
  agent; Groq-backed RCA proposal (VCR-cassette tested).
- `PLCBridge` (Modbus TCP client) + `VirtualModbusServer`; `APEX_PLC_MODE`
  defaults to `simulation`; trigger engine (3-consecutive or yield-drift >15%)
  → `interrupt()` checkpoint → supervisor resume → `resolve_ticket()` dispatch.
- Dual-mode persistence: Neon Postgres with zero-config SQLite fallback;
  `X-API-KEY` gateway auth; append-only-by-convention `AuditLog` (honestly
  documented as NOT tamper-evident — no trigger/hash chain).

**Result.** 43.36 ms/frame local benchmark; fail-closed detector + PLC
simulation-default covered by 6 Phase-2 tests; audit honestly scopes what the
log does and does not prove.

## 2. System-design Q&A

**Q1: Why BM25 instead of vector RAG for SOPs? (ADR-001)**
6 short SOP files with exact terminology (IPC-A-610 Class 3, TAL/PWI) need
exact-match precision, not semantic drift; BM25 is deterministic, debuggable,
and needs no embed service on the edge. Vectors win past hundreds of long
documents — not this corpus.

**Q2: Why fail-closed on detector error? (ADR-004)**
A detector that fails open ships defects; a detector that fails closed stops
the line — expensive but safe, and the annoyance is the signal to fix the
model. Fail-open optimises throughput over correctness, which inverts the
factory's actual loss function (a field escape costs orders more than downtime).

**Q3: Why simulation-default PLC mode?**
A default deployment that touches hardware is a safety hazard: `hardware` mode
requires explicit opt-in on an approved line. Simulation-default also makes
every CI run exercise the full dispatch path without hardware. The critical
invariant: a failed hardware write is NEVER reported as simulated success.

**Q4: SPOF / why not tamper-evident audit?**
Single gateway + single PLC link per line; Neon→SQLite fallback covers data
plane, not actuation. The AuditLog is append-only by convention only — claiming
tamper-evidence without triggers/hash-chain would be dishonest; regulated lines
need WORM storage, which is scoped future work, not a silent gap.

## 3. Live-demo script (5 steps)

```bash
# 1. Run offline (simulation PLC, SQLite fallback — no hardware, no cloud)
APEX_PLC_MODE=simulation uvicorn src.backend.main:app --port 8000
# 2. Ingest a sample frame -> defect telemetry (try data/sample_pcbs/mouse_bite_1.jpg)
curl -s http://localhost:8000/inspect -H 'X-API-KEY: <key>' -F 'image=@data/sample_pcbs/mouse_bite_1.jpg'
# 3. Fire the trigger path (3 consecutive defects) -> incident + SOP-grounded RCA draft
curl -s http://localhost:8000/incidents/latest
# 4. Supervisor approves at the HITL checkpoint
curl -s http://localhost:8000/incidents/<id>/approve -X POST -H 'X-API-KEY: <key>' -d '{"approved":true}'
# 5. Verify dispatch hit the (virtual) PLC + audit row, and benchmark honesty
curl -s http://localhost:8000/audit/latest && python scripts/benchmark_inference.py
```

| Step | URL | Expected |
|---|---|---|
| 2 | `POST /inspect` | BBox + class + confidence telemetry |
| 3 | `GET /incidents/latest` | RCA draft citing SOP file/section |
| 4 | `POST /incidents/{id}/approve` | Dispatch only after approval |
| 5 | `GET /audit/latest` + benchmark | Audit row; ~43 ms/frame Apple M-series, not a prod promise |
