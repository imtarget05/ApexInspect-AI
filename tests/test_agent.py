import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.rag import SOPRetriever
from src.agent.graph import QualityIncidentAgent

class TestAgentPipeline(unittest.TestCase):
    """Unit tests for SOP RAG retrieval and Agentic incident resolution."""

    def setUp(self):
        sops_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/sops"))
        self.retriever = SOPRetriever(sops_dir=sops_dir)
        self.agent = QualityIncidentAgent(sops_dir=sops_dir)

    def test_sop_retriever_loads_documents(self):
        """Retriever must discover and load all 3 sample SOP markdown manuals."""
        self.assertGreaterEqual(len(self.retriever.documents), 3)
        doc_ids = [d["id"] for d in self.retriever.documents]
        self.assertTrue(any("SOP-SMT-001" in doc_id for doc_id in doc_ids))

    def test_sop_search_finds_solder_bridge(self):
        """Searching for 'short_circuit' must return SOP-SMT-001 with high score."""
        results = self.retriever.search("short_circuit", top_k=1)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("SOP-SMT-001", results[0]["sop_id"])

    def test_agent_resolves_critical_short_circuit(self):
        """Agent must recommend HALT_LINE and cite SOP-SMT-001 when 3 short circuits occur."""
        result = self.agent.run(defect_class="short_circuit", consecutive_count=3, line_id="SMT-LINE-01")
        self.assertEqual(result["proposed_action"], "HALT_LINE")
        self.assertTrue(result["requires_hitl"])
        self.assertIn("SOP-SMT-001", str(result["sop_citations"]))
        self.assertIn("RCA", result["rca_analysis"])

if __name__ == "__main__":
    unittest.main()
