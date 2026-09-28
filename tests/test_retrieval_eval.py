"""Tests for retrieval-level BM25 evaluation.

The agent-level harness (``test_eval_harness.py``) never measures the index
itself. These tests pin it: real-retriever quality gates plus unit checks of
the metric functions with fake retrievers, so a future metric bug cannot
silently inflate scores.

Uses unittest (not bare pytest) to match the rest of tests/.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.retrieval_dataset import RETRIEVAL_QUERIES
from eval.retrieval_eval import (
    evaluate_retriever,
    hit_at_1,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.agent.rag import SOPRetriever

_SOPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sops"))


class _FakeRetriever:
    """Retriever double returning scripted sop_id lists per query."""

    def __init__(self, ranked_by_query=None, default=()):
        self.ranked_by_query = ranked_by_query or {}
        self.default = list(default)

    def search(self, query, top_k=3):
        ranked = self.ranked_by_query.get(query, self.default)
        return [{"sop_id": sid, "score": 1.0, "snippet": ""} for sid in ranked[:top_k]]


class TestMetricFunctions(unittest.TestCase):
    def test_perfect_ranking_scores_one(self):
        ranked = ["SOP-SMT-001-solder-bridge", "SOP-SMT-004-ipc610-solder-criteria"]
        relevant = ["SOP-SMT-001-solder-bridge"]
        self.assertEqual(recall_at_k(ranked, relevant, 1), 1.0)
        self.assertEqual(recall_at_k(ranked, relevant, 3), 1.0)
        self.assertEqual(precision_at_k(ranked, relevant, 1), 1.0)
        self.assertEqual(reciprocal_rank(ranked, relevant), 1.0)
        self.assertEqual(hit_at_1(ranked, relevant), 1.0)

    def test_second_rank_scores_half_rr(self):
        ranked = ["SOP-SMT-004-ipc610-solder-criteria", "SOP-SMT-001-solder-bridge"]
        relevant = ["SOP-SMT-001-solder-bridge"]
        self.assertEqual(recall_at_k(ranked, relevant, 1), 0.0)
        self.assertEqual(recall_at_k(ranked, relevant, 2), 1.0)
        self.assertAlmostEqual(reciprocal_rank(ranked, relevant), 0.5)
        self.assertEqual(hit_at_1(ranked, relevant), 0.0)

    def test_multi_relevant_partial_recall(self):
        ranked = ["SOP-SMT-002-missing-component", "SOP-SMT-003-line-halt-safety"]
        relevant = ["SOP-SMT-002-missing-component", "SOP-SMT-006-pick-place-nozzle-maintenance"]
        self.assertAlmostEqual(recall_at_k(ranked, relevant, 2), 0.5)
        self.assertAlmostEqual(precision_at_k(ranked, relevant, 2), 0.5)

    def test_no_hit_scores_zero(self):
        ranked = ["SOP-SMT-004-ipc610-solder-criteria"]
        relevant = ["SOP-SMT-001-solder-bridge"]
        self.assertEqual(recall_at_k(ranked, relevant, 3), 0.0)
        self.assertEqual(precision_at_k(ranked, relevant, 3), 0.0)
        self.assertEqual(reciprocal_rank(ranked, relevant), 0.0)
        self.assertEqual(hit_at_1(ranked, relevant), 0.0)
        self.assertEqual(recall_at_k([], relevant, 3), 0.0)

    def test_short_and_full_sop_ids_match(self):
        """Ground truth may use 'SOP-SMT-001' or the full file stem — both count."""
        ranked = ["SOP-SMT-001-solder-bridge"]
        self.assertEqual(recall_at_k(ranked, ["SOP-SMT-001"], 1), 1.0)
        self.assertEqual(recall_at_k(["SOP-SMT-001"], ranked, 1), 1.0)


class TestEvaluateRetrieverWithFakes(unittest.TestCase):
    def test_perfect_fake_retriever_scores_one(self):
        dataset = [
            {"id": "q1", "query": "q1", "relevant": ["SOP-SMT-001-solder-bridge"]},
            {"id": "q2", "query": "q2", "relevant": ["SOP-SMT-002-missing-component"]},
        ]
        fake = _FakeRetriever({
            "q1": ["SOP-SMT-001-solder-bridge"],
            "q2": ["SOP-SMT-002-missing-component"],
        })
        report = evaluate_retriever(fake, dataset, top_k=3)
        self.assertEqual(report.total, 2)
        self.assertEqual(report.recall_at(3), 1.0)
        self.assertEqual(report.mrr, 1.0)
        self.assertEqual(report.hit_at_1, 1.0)
        self.assertIn("Recall@3", report.to_text())
        self.assertEqual(report.to_dict()["summary"]["recall@3"], 1.0)

    def test_empty_fake_retriever_scores_zero(self):
        dataset = [{"id": "q1", "query": "q1", "relevant": ["SOP-SMT-001-solder-bridge"]}]
        report = evaluate_retriever(_FakeRetriever(), dataset, top_k=3)
        self.assertEqual(report.recall_at(3), 0.0)
        self.assertEqual(report.mrr, 0.0)
        self.assertEqual(report.hit_at_1, 0.0)


class TestDatasetIntegrity(unittest.TestCase):
    def test_dataset_size_in_range(self):
        self.assertGreaterEqual(len(RETRIEVAL_QUERIES), 12, "cần >= 12 queries")
        self.assertLessEqual(len(RETRIEVAL_QUERIES), 15, "cần <= 15 queries")

    def test_query_ids_unique(self):
        ids = [q["id"] for q in RETRIEVAL_QUERIES]
        self.assertEqual(len(ids), len(set(ids)), "id trùng làm mất khả năng truy vết query")

    def test_each_query_has_one_or_two_relevant(self):
        for q in RETRIEVAL_QUERIES:
            self.assertTrue(q.get("query"), f"{q['id']} thiếu query text")
            self.assertGreaterEqual(len(q.get("relevant", [])), 1, f"{q['id']} thiếu ground truth")
            self.assertLessEqual(len(q.get("relevant", [])), 2, f"{q['id']} quá 2 SOP đúng")

    def test_ground_truth_resolves_to_real_files(self):
        import re
        stems = {f.split(".")[0] for f in os.listdir(_SOPS_DIR) if f.endswith(".md")}
        file_cores = set()
        for s in stems:
            m = re.search(r"SOP-SMT-\d+", s)
            self.assertIsNotNone(m, f"file SOP lạ: {s}")
            file_cores.add(m.group(0).upper())
        for q in RETRIEVAL_QUERIES:
            for sid in q["relevant"]:
                m = re.search(r"SOP-SMT-\d+", str(sid))
                self.assertIsNotNone(m, f"{q['id']}: ground truth lạ {sid}")
                self.assertIn(m.group(0).upper(), file_cores, f"{q['id']}: {sid} không khớp file SOP nào")

    def test_all_six_sops_covered(self):
        import re
        covered = set()
        for q in RETRIEVAL_QUERIES:
            for sid in q["relevant"]:
                covered.add(re.search(r"SOP-SMT-\d+", str(sid)).group(0).upper())
        self.assertEqual(len(covered), 6, f"dataset phải phủ cả 6 SOP, thiếu: {covered}")


class TestRealRetrieverQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = SOPRetriever(sops_dir=_SOPS_DIR)
        cls.report = evaluate_retriever(cls.retriever, RETRIEVAL_QUERIES, top_k=3)

    def test_recall_at_3_meets_gate(self):
        self.assertGreaterEqual(
            self.report.recall_at(3), 0.7,
            f"Recall@3={self.report.recall_at(3):.3f} dưới gate 0.7\n{self.report.to_text()}",
        )

    def test_mrr_above_zero(self):
        self.assertGreater(
            self.report.mrr, 0.0,
            f"MRR={self.report.mrr:.3f} — retriever không trả về SOP đúng nào",
        )

    def test_search_result_shape(self):
        hits = self.retriever.search(RETRIEVAL_QUERIES[0]["query"], top_k=2)
        self.assertTrue(hits, "search trả về rỗng cho query đầu tiên")
        for hit in hits:
            for key in ("sop_id", "filename", "score", "snippet"):
                self.assertIn(key, hit, f"kết quả search thiếu key {key}")


if __name__ == "__main__":
    unittest.main()
