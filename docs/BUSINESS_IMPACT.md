# Business Impact — ApexInspect-AI (Smart Factory Vision & Quality Agent)

Domain facts: SMT/PCBA edge vision (YOLOv8 ONNX), 3-consecutive/Yield-drift
trigger, BM25 SOP RAG (IPC-A-610), LangGraph HITL approval, Modbus TCP PLC
dispatch (simulation by default), append-only-by-convention AuditLog (not
tamper-evident). See `README.md`, `benchmark_results.json`.

## Problem (cost of status quo)

Escaped defects and late line-stops are expensive (~$12k/hr line-stop,
ESTIMATE — plan assumption, not a measured customer figure). Manual inspection
misses small/overlapping defects; root-cause lookup in SOP binders is slow.

## Solution (what the system does)

`Inspect → Detect → Diagnose → Propose → Human Approve → PLC Dispatch & MES Sync`:
ONNX edge inference → trigger engine → SOP-grounded RCA draft → supervisor
approval → PLC actuation + MES sync, every step audit-logged.

## Impact

| Metric | Before | After | How measured |
|---|---|---|---|
| False-negative rate | 4.2% | 0.3% | ESTIMATE — plan target; NOT demonstrated: committed model scores zero detections on all 13 real frames in `data/sample_pcbs/` (see README scope note). Pending retrain + real-frame eval. |
| Line-stop cost exposure | $12k/hr at risk | reduced via early trigger | ESTIMATE — plan assumption, not a measured customer figure. |
| ONNX latency (synthetic frames, M-series macOS) | — | avg 43.36 ms / 23.1 FPS | MEASURED — `benchmark_results.json` (2026-09-17), 20 runs + 5 warmup; synthetic frames only, not production. |
| IoU math micro-op (20k iters, local) | — | mean 0.028 ms, p95 0.002 ms | MEASURED by `scripts/bench_apex.py` (stdlib math baseline; `src.vision.patching` unavailable without cv2), this machine 2026-09-27. |
| Debounce (3-consecutive) micro-op (2k iters, local) | — | mean 0.89 ms, p95 0.28 ms | MEASURED by `scripts/bench_apex.py`, same conditions. |

## Guardrails / SLO links

- `APEX_PLC_MODE=simulation` default (no real actuation); HITL interrupt/resume;
  per-frame `inference_time_ms` in DB; yield/defect metrics via API.
- SLOs: `observability/slo.yaml` (job `apexinspect` :7860).
- Reproduce latency: `python scripts/benchmark_inference.py`,
  `python scripts/check_benchmark_target.py` (target ≤35 ms cloud 2-vCPU, unverified).

## Reproduce

```bash
cd ApexInspect-AI
python3 scripts/bench_apex.py
python scripts/benchmark_inference.py
python -m pytest tests/ -q
```
