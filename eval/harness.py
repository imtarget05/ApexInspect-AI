"""Evaluation harness for the ApexInspect agent.

Why this exists
---------------
The agent's decision has a physical consequence: ``HALT_LINE`` stops a
production line, ``ROUTE_REWORK`` diverts product. A wrong answer is not a bad
sentence — it is a scrapped shift. Unit tests assert the graph *plumbing*; this
harness asserts the agent actually *decides correctly and stays grounded* on
scenarios a QA engineer would recognise.

Four independent checks per case, because they fail in different ways:

1. ``decision``   — does the proposed action match the golden one?
2. ``grounding``  — is the SOP the right one cited? An answer grounded in the
                    wrong manual is wrong even when the conclusion is right.
3. ``content``    — does the RCA name the actual mechanism (stencil, nozzle)?
4. ``hallucination`` — does it claim something the SOPs never say? This check
                    exists to FAIL LOUD: a single hallucinated execution claim
                    must never be averaged away into a passing score.

Scoring is deliberately not a single number. ``decision_accuracy`` can sit at
100% while ``hallucination_rate`` is 20%; reporting only the first would hide
exactly the failure that matters. The caller decides the gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from eval.dataset import GOLDEN_CASES


@dataclass
class CaseResult:
    """Outcome of scoring one golden case."""

    case_id: str
    severity: str
    decision_ok: bool
    grounding_ok: bool
    content_ok: bool
    hallucination_free: bool
    proposed_action: str = ""
    citations: List[str] = field(default_factory=list)
    rca: str = ""
    failures: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.decision_ok
            and self.grounding_ok
            and self.content_ok
            and self.hallucination_free
        )


@dataclass
class EvalReport:
    """Aggregate of a full run. No single score — see module docstring."""

    results: List[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def decision_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.decision_ok) / len(self.results)

    @property
    def grounding_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.grounding_ok) / len(self.results)

    @property
    def hallucination_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if not r.hallucination_free) / len(self.results)

    @property
    def critical_failures(self) -> List[CaseResult]:
        """Cases where a wrong answer would endanger a production line."""
        return [r for r in self.results if r.severity == "CRITICAL" and not r.passed]

    def gate(self, min_decision_accuracy: float = 1.0, max_hallucination_rate: float = 0.0) -> Dict[str, Any]:
        """Return pass/fail against a release gate.

        Defaults are strict on purpose: 100% decision accuracy and zero
        hallucination. A safety-relevant agent that is "usually right" is not
        shippable, and a single hallucinated execution claim is disqualifying
        regardless of the aggregate.
        """
        checks = {
            "decision_accuracy": (self.decision_accuracy, ">=", min_decision_accuracy),
            "hallucination_rate": (self.hallucination_rate, "<=", max_hallucination_rate),
            "critical_cases": (len(self.critical_failures), "<=", 0),
        }
        failures = [
            f"{name}={value:.2f} (expected {op} {threshold})"
            for name, (value, op, threshold) in checks.items()
            if not _satisfies(value, op, threshold)
        ]
        return {
            "passed": not failures,
            "failures": failures,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "decision_accuracy": round(self.decision_accuracy, 4),
                "grounding_accuracy": round(self.grounding_accuracy, 4),
                "hallucination_rate": round(self.hallucination_rate, 4),
            },
        }

    def to_text(self) -> str:
        lines = [
            f"ApexInspect agent evaluation — {self.passed}/{self.total} cases passed",
            f"  decision accuracy    : {self.decision_accuracy:.1%}",
            f"  grounding accuracy   : {self.grounding_accuracy:.1%}",
            f"  hallucination rate   : {self.hallucination_rate:.1%}",
        ]
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{mark}] {r.case_id} ({r.severity}) -> {r.proposed_action or 'n/a'}")
            for failure in r.failures:
                lines.append(f"         - {failure}")
        if self.critical_failures:
            lines.append("  !! CRITICAL cases failed — a production line decision was unsafe.")
        return "\n".join(lines)


def _satisfies(value: float, op: str, threshold: float) -> bool:
    return value >= threshold if op == ">=" else value <= threshold


def _norm(text: Any) -> str:
    return str(text or "").lower()


def score_case(case: Dict[str, Any], output: Dict[str, Any]) -> CaseResult:
    """Score one agent output against one golden case.

    ``output`` is the dict returned by ``QualityIncidentAgent.run`` — the same
    shape the production code path produces, so the harness measures the real
    thing rather than a special eval-only branch.
    """
    proposed = str(output.get("proposed_action", ""))
    citations = [str(c) for c in (output.get("sop_citations") or [])]
    rca = str(output.get("rca_analysis", ""))
    rca_norm = _norm(rca)
    joined_citations = " ".join(citations)

    failures: List[str] = []

    decision_ok = proposed == case["expect_action"]
    if not decision_ok:
        failures.append(f"decision: expected {case['expect_action']}, got {proposed or 'n/a'}")

    # A case with `expect_sop: None` only requires that SOMETHING sensible is
    # cited, not a specific manual.
    if case.get("expect_sop"):
        grounding_ok = case["expect_sop"] in joined_citations
    else:
        grounding_ok = bool(citations)
    if not grounding_ok:
        expected = case.get("expect_sop") or "ít nhất một SOP"
        failures.append(f"grounding: expected citation {expected}, got {citations or 'none'}")

    missing = [term for term in case.get("must_contain", []) if _norm(term) not in rca_norm]
    content_ok = not missing
    if missing:
        failures.append(f"content: RCA thiếu căn cứ {missing}")

    present = [term for term in case.get("must_not_contain", []) if _norm(term) in rca_norm]
    hallucination_free = not present
    if present:
        failures.append(f"hallucination: RCA khẳng định sai {present}")

    return CaseResult(
        case_id=case["id"],
        severity=case["severity"],
        decision_ok=decision_ok,
        grounding_ok=grounding_ok,
        content_ok=content_ok,
        hallucination_free=hallucination_free,
        proposed_action=proposed,
        citations=citations,
        rca=rca,
        failures=failures,
    )


def run_evaluation(agent, cases: List[Dict[str, Any]] | None = None) -> EvalReport:
    """Run the agent over the golden set and score every outcome.

    ``agent`` is any object exposing ``run(defect_class=, consecutive_count=,
    line_id=)`` — the real ``QualityIncidentAgent`` in production use, or a fake
    in unit tests of the harness itself.

    Every case gets a fresh ``thread_id``. The agent persists incident state per
    thread; reusing one id across cases would let an earlier approval leak into a
    later scenario and quietly corrupt the scores.
    """
    report = EvalReport()
    for case in cases if cases is not None else GOLDEN_CASES:
        try:
            output = agent.run(
                defect_class=case["defect_class"],
                consecutive_count=case["count"],
                line_id=case["line_id"],
                thread_id=f"eval-{case['id']}",
            )
        except Exception as exc:  # a crash is a failed case, not a failed run
            report.results.append(
                CaseResult(
                    case_id=case["id"],
                    severity=case["severity"],
                    decision_ok=False,
                    grounding_ok=False,
                    content_ok=False,
                    hallucination_free=False,
                    failures=[f"agent raised {type(exc).__name__}: {exc}"],
                )
            )
            continue
        report.results.append(score_case(case, output))
    return report
