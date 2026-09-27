"""Agent evaluation for ApexInspect-AI.

Separated from ``tests/`` on purpose: tests assert that the code runs, the eval
harness asserts that the agent is *right*. A regression in decision quality must
fail the release gate, not quietly sit inside a green unit-test run.

Usage::

    python -m eval.run                 # run the golden set, print the report
    python -m eval.run --json          # machine-readable, for CI
    python -m eval.run --strict        # non-zero exit unless the gate passes
"""
from eval.dataset import GOLDEN_CASES, critical_cases
from eval.harness import CaseResult, EvalReport, run_evaluation, score_case

__all__ = [
    "GOLDEN_CASES",
    "critical_cases",
    "CaseResult",
    "EvalReport",
    "run_evaluation",
    "score_case",
]
