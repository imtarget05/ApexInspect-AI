# ADR-004: Fail-Closed Detector with Groq-Backed RCA

- **Status:** Accepted
- **Date:** 2026-09-27

## Context

Two coupled risks: (1) the vision detector can error (bad frame, corrupt
weights, OOM) mid-shift — the line must not silently pass uninspected boards;
(2) RCA needs LLM reasoning over SOPs, but live-LLM tests are flaky and
leak-secrets prone in CI.

## Decision

Detector errors fail CLOSED: any inference exception quarantines the board,
halts auto-disposition, and raises an incident instead of a pass. RCA proposals
come from the Groq-backed agent over BM25 SOP context, tested via VCR
cassettes (`test_rca_groq_vcr`) so CI replays recorded LLM traffic with zero
live calls and zero secrets.

## Consequences

- Positive: no silent escapes on model failure; RCA path is deterministic in CI
  and live only where credentials exist (guarded skip otherwise).
- Negative: fail-closed stops the line on transient GPU/frame glitches —
  mitigated by retry-once-then-quarantine; VCR cassettes must be re-recorded
  when prompts change.

## Alternatives

- Fail-open with low-confidence pass: maximises throughput, but a field escape
  (shipped defect) costs orders more than downtime; rejected on loss-function
  grounds.
- Live-LLM RCA tests: realistic, but flaky, slow, and secret-bearing; VCR
  replay keeps the same assertions without the same costs.
