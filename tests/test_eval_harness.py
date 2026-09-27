"""Tests for the evaluation harness itself.

A gate that cannot fail is not a gate. These tests inject deliberately broken
agents and assert the harness CATCHES them — otherwise a future refactor could
make the harness return 100% forever and nobody would notice until an incident.

Uses unittest (not bare pytest) to match the rest of tests/.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.dataset import GOLDEN_CASES, critical_cases
from eval.harness import run_evaluation, score_case


class _FakeAgent:
    """Agent double returning a scripted output per defect_class."""

    def __init__(self, output_factory):
        self.output_factory = output_factory
        self.calls = []

    def run(self, defect_class, consecutive_count, line_id, thread_id=None):
        self.calls.append({
            "defect_class": defect_class,
            "consecutive_count": consecutive_count,
            "line_id": line_id,
            "thread_id": thread_id,
        })
        return self.output_factory(defect_class, consecutive_count)


class _CrashingAgent:
    def run(self, **kwargs):
        raise RuntimeError("PLC link down")


class TestScoreCase(unittest.TestCase):
    """Scoring logic: each check must fail for the right reason."""

    def test_perfect_output_scores_all_four_checks(self):
        case = GOLDEN_CASES[0]
        output = {
            "proposed_action": case["expect_action"],
            "sop_citations": [case["expect_sop"]],
            "rca_analysis": "Stencil bị đọng cặn thiếc hàn dư thừa gây hàn chập liên tiếp.",
        }
        result = score_case(case, output)
        self.assertTrue(result.decision_ok)
        self.assertTrue(result.grounding_ok)
        self.assertTrue(result.content_ok)
        self.assertTrue(result.hallucination_free)
        self.assertTrue(result.passed)
        self.assertEqual(result.failures, [])

    def test_wrong_decision_is_caught(self):
        case = GOLDEN_CASES[0]
        result = score_case(case, {
            "proposed_action": "ROUTE_REWORK",  # wrong: case expects HALT_LINE
            "sop_citations": [case["expect_sop"]],
            "rca_analysis": "stencil hàn",
        })
        self.assertFalse(result.decision_ok)
        self.assertFalse(result.passed)
        self.assertTrue(any("decision" in f for f in result.failures))

    def test_right_answer_wrong_grounding_still_fails(self):
        """Đúng kết luận nhưng sai SOP vẫn phải fail — căn cứ sai thì sai."""
        case = GOLDEN_CASES[0]
        result = score_case(case, {
            "proposed_action": case["expect_action"],
            "sop_citations": ["SOP-SMT-006"],  # sai manual
            "rca_analysis": "stencil hàn",
        })
        self.assertTrue(result.decision_ok)
        self.assertFalse(result.grounding_ok)
        self.assertFalse(result.passed)

    def test_hallucination_fails_case(self):
        case = GOLDEN_CASES[0]
        result = score_case(case, {
            "proposed_action": case["expect_action"],
            "sop_citations": [case["expect_sop"]],
            "rca_analysis": "Stencil hàn — hệ thống tự động tiếp tục không cần can thiệp.",
        })
        self.assertTrue(result.decision_ok)
        self.assertFalse(result.hallucination_free)
        self.assertFalse(result.passed)

    def test_missing_content_fails_case(self):
        case = GOLDEN_CASES[0]
        result = score_case(case, {
            "proposed_action": case["expect_action"],
            "sop_citations": [case["expect_sop"]],
            "rca_analysis": "Lỗi chất lượng mạ không xác định.",  # không có "stencil"/"hàn"
        })
        self.assertFalse(result.content_ok)
        self.assertFalse(result.passed)

    def test_missing_sop_when_not_required_still_needs_some_citation(self):
        case = [c for c in GOLDEN_CASES if c.get("expect_sop") is None][0]
        no_citation = score_case(case, {
            "proposed_action": case["expect_action"],
            "sop_citations": [],
            "rca_analysis": "Rework thủ công.",
        })
        self.assertFalse(no_citation.grounding_ok, "không trích SOP nào vẫn phải fail grounding")
        with_citation = score_case(case, {
            "proposed_action": case["expect_action"],
            "sop_citations": ["SOP-SMT-002"],
            "rca_analysis": "Rework thủ công.",
        })
        self.assertTrue(with_citation.grounding_ok)


class TestHarnessCatchesBrokenAgents(unittest.TestCase):
    """The gate must reject bad agents — these are the tests that matter."""

    def test_always_halt_agent_fails_the_gate(self):
        """Agent that halts everything is the classic failure mode."""

        def factory(defect_class, count):
            return {
                "proposed_action": "HALT_LINE",
                "sop_citations": ["SOP-SMT-001"],
                "rca_analysis": "Stencil hàn bị lỗi liên tiếp.",
            }

        report = run_evaluation(_FakeAgent(factory))
        verdict = report.gate()
        self.assertFalse(verdict["passed"], "agent dừng mọi dây chuyền phải bị gate chặn")
        self.assertLess(report.decision_accuracy, 1.0)
        self.assertTrue(report.critical_failures, "case CRITICAL sai phải được báo")

    def test_hallucinating_agent_fails_despite_correct_decisions(self):
        """100% quyết định đúng nhưng bịa đặt thì vẫn phải fail."""
        cases = GOLDEN_CASES[:2]

        def factory(defect_class, count):
            return {
                "proposed_action": "HALT_LINE" if count >= 3 else "ROUTE_REWORK",
                "sop_citations": ["SOP-SMT-001"],
                "rca_analysis": "Hệ thống tự động tiếp tục, stencil hàn không cần can thiệp.",
            }

        report = run_evaluation(_FakeAgent(factory), cases)
        self.assertEqual(report.decision_accuracy, 1.0, "test này cần quyết định đúng trước")
        self.assertGreater(report.hallucination_rate, 0.0)
        self.assertFalse(report.gate()["passed"], "bịa đặt phải fail gate dù quyết định đúng")

    def test_crashing_agent_yields_failed_cases_not_a_crashed_run(self):
        report = run_evaluation(_CrashingAgent(), GOLDEN_CASES[:2])
        self.assertEqual(report.total, 2)
        self.assertEqual(report.passed, 0)
        self.assertFalse(report.gate()["passed"])
        self.assertTrue(any("raised" in f for f in report.results[0].failures))

    def test_report_metrics_are_independent(self):
        """decision_accuracy=100% phải có thể đi cùng hallucination_rate>0."""
        def factory(defect_class, count):
            return {
                "proposed_action": "HALT_LINE" if count >= 3 else "ROUTE_REWORK",
                "sop_citations": ["SOP-SMT-001"],
                "rca_analysis": "stencil hàn — tự động tiếp tục.",
            }

        report = run_evaluation(_FakeAgent(factory), GOLDEN_CASES[:2])
        self.assertEqual(report.decision_accuracy, 1.0)
        self.assertGreater(report.hallucination_rate, 0.0)


class TestRunEvaluationHygiene(unittest.TestCase):
    def test_each_case_uses_a_fresh_thread_id(self):
        agent = _FakeAgent(lambda d, c: {
            "proposed_action": "ROUTE_REWORK",
            "sop_citations": ["SOP-SMT-001"],
            "rca_analysis": "Rework thủ công.",
        })
        run_evaluation(agent, GOLDEN_CASES)
        thread_ids = [c["thread_id"] for c in agent.calls]
        self.assertEqual(len(thread_ids), len(set(thread_ids)), "mỗi case cần thread riêng")
        self.assertTrue(all(t and t.startswith("eval-") for t in thread_ids))


class TestDatasetIntegrity(unittest.TestCase):
    def test_dataset_has_critical_cases(self):
        """Không có case CRITICAL thì gate mất khả năng chặn lỗi nghiêm trọng."""
        self.assertTrue(critical_cases(), "dataset phải có ít nhất một case CRITICAL")

    def test_case_ids_unique(self):
        ids = [c["id"] for c in GOLDEN_CASES]
        self.assertEqual(len(ids), len(set(ids)), "id trùng làm mất khả năng truy vết case")

    def test_every_case_declares_its_expectations(self):
        for case in GOLDEN_CASES:
            self.assertIn("expect_action", case, f"{case['id']} thiếu expect_action")
            self.assertIn(case["expect_action"], {"HALT_LINE", "ROUTE_REWORK"})
            self.assertIn("severity", case)
            self.assertTrue(case.get("must_contain"), f"{case['id']} không có tiêu chí căn cứ")


if __name__ == "__main__":
    unittest.main()
