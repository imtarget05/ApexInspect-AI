"""CLI entry point: ``python -m eval.run``.

Keeps the harness runnable the same way in CI, in a pre-release check, and on
a laptop — a gate nobody runs is not a gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

# The agent resolves the SOP corpus relative to the working directory, and the
# checkpointer writes SQLite next to it. Point both at a throwaway directory so
# an eval run never touches the developer's `factory.db` or a production DB.
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SOPS = os.path.join(_HERE, "data", "sops")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the ApexInspect agent against the golden set.")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--strict", action="store_true", help="exit 1 when the release gate fails")
    parser.add_argument("--sops-dir", default=_SOPS, help="SOP corpus directory")
    parser.add_argument(
        "--min-decision-accuracy",
        type=float,
        default=1.0,
        help="gate threshold for decision accuracy (default: 1.0 — every case must be right)",
    )
    parser.add_argument(
        "--report",
        help="also write the JSON report to this path (CI uploads it as an artifact)",
    )
    args = parser.parse_args(argv)

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    from src.agent.graph import QualityIncidentAgent
    from eval.harness import run_evaluation

    # Isolate the incident checkpointer: a real run must not append to a
    # developer's local incident history.
    workdir = tempfile.mkdtemp(prefix="apex-eval-")
    os.environ["APEX_CHECKPOINT_DB"] = os.path.join(workdir, "eval-checkpoints.db")

    agent = QualityIncidentAgent(sops_dir=args.sops_dir)
    report = run_evaluation(agent)
    verdict = report.gate(min_decision_accuracy=args.min_decision_accuracy)

    payload = {
        "gate": verdict,
        "summary": {
            "total": report.total,
            "passed": report.passed,
            "decision_accuracy": report.decision_accuracy,
            "grounding_accuracy": report.grounding_accuracy,
            "hallucination_rate": report.hallucination_rate,
        },
        "cases": [
            {
                "id": r.case_id,
                "severity": r.severity,
                "passed": r.passed,
                "proposed_action": r.proposed_action,
                "citations": r.citations,
                "failures": r.failures,
            }
            for r in report.results
        ],
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(report.to_text())
        if verdict["passed"]:
            print("\nGATE: PASS")
        else:
            print("\nGATE: FAIL")
            for failure in verdict["failures"]:
                print(f"  - {failure}")

    # Written before the exit code so a failing gate still leaves evidence.
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\nReport written to {args.report}")

    if args.strict and not verdict["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
