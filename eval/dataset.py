"""Golden dataset for the ApexInspect agent evaluation harness.

Each case is a real incident scenario with the action a competent QA engineer
would have chosen, derived from the SOP corpus in ``data/sops/``. The dataset is
the fixed reference the harness scores against — changing a value here is a
deliberate, reviewable act, not an accident of model output.

Schema per case
---------------
``id``            stable identifier, referenced by the report.
``defect_class``  detector label fed to the agent.
``count``         consecutive defect occurrences (drives the HALT_LINE rule).
``line_id``       production line identifier.
``expect_action`` ``HALT_LINE`` or ``ROUTE_REWORK`` — the correct decision.
``expect_sop``    SOP the answer must cite (grounding check).
``must_contain``  substrings the RCA must mention to count as grounded.
``must_not_contain``  strings that would indicate a hallucinated claim.
``severity``      CRITICAL / MODERATE, used for the safety gate.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Severity ordering is intentional: CRITICAL cases are safety-relevant and a
# miss on them must fail the whole run, not just lower an average score.
SEVERITY_ORDER = {"CRITICAL": 2, "MODERATE": 1, "LOW": 0}

GOLDEN_CASES: List[Dict[str, Any]] = [
    {
        "id": "short_circuit_3x_halt",
        "defect_class": "short_circuit",
        "count": 3,
        "line_id": "SMT-LINE-01",
        "expect_action": "HALT_LINE",
        "expect_sop": "SOP-SMT-001",
        "must_contain": ["stencil", "hàn"],
        "must_not_contain": ["tự động tiếp tục", "không cần can thiệp"],
        "severity": "CRITICAL",
    },
    {
        "id": "short_circuit_1x_rework",
        "defect_class": "short_circuit",
        "count": 1,
        "line_id": "SMT-LINE-01",
        "expect_action": "ROUTE_REWORK",
        "expect_sop": "SOP-SMT-001",
        "must_contain": ["hàn"],
        "must_not_contain": ["HALT_LINE đã thực thi"],
        "severity": "MODERATE",
    },
    {
        "id": "missing_component_3x_halt",
        "defect_class": "missing_component",
        "count": 3,
        "line_id": "SMT-LINE-02",
        "expect_action": "HALT_LINE",
        "expect_sop": "SOP-SMT-002",
        "must_contain": ["linh kiện"],
        "must_not_contain": ["tự động tiếp tục"],
        "severity": "CRITICAL",
    },
    {
        "id": "mouse_bite_single_rework",
        "defect_class": "mouse_bite",
        "count": 1,
        "line_id": "SMT-LINE-02",
        "expect_action": "ROUTE_REWORK",
        "expect_sop": None,
        "must_contain": ["Rework"],
        "must_not_contain": ["tự động tiếp tục"],
        "severity": "MODERATE",
    },
    {
        "id": "spurious_copper_cascade_halt",
        "defect_class": "spurious_copper",
        "count": 4,
        "line_id": "SMT-LINE-01",
        # Not a blocking class in the governance rule → REWORK even at count 4.
        # This is the case a model is most likely to get wrong: it pattern-matches
        # "count >= 3 → halt" instead of applying the class-specific rule.
        "expect_action": "ROUTE_REWORK",
        "expect_sop": None,
        "must_contain": ["Rework"],
        "must_not_contain": ["tự động tiếp tục"],
        "severity": "CRITICAL",
    },
    {
        "id": "open_circuit_two_rework",
        "defect_class": "open_circuit",
        "count": 2,
        "line_id": "SMT-LINE-03",
        "expect_action": "ROUTE_REWORK",
        "expect_sop": None,
        "must_contain": ["Rework"],
        "must_not_contain": ["tự động tiếp tục"],
        "severity": "MODERATE",
    },
]


def critical_cases() -> List[Dict[str, Any]]:
    """Cases where a wrong answer endangers a production line."""
    return [c for c in GOLDEN_CASES if c["severity"] == "CRITICAL"]


def by_id(case_id: str) -> Dict[str, Any]:
    for case in GOLDEN_CASES:
        if case["id"] == case_id:
            return case
    raise KeyError(f"unknown golden case: {case_id}")
