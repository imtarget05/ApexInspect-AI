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

    def test_decoupled_subagents_execution(self):
        """Verify each sub-agent can be executed independently."""
        from src.agent.subagents import SOPResearchAgent, RCAAnalysisAgent, InterventionGovernanceAgent
        from src.agent.graph import QualityIncidentOrchestrator

        sops_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/sops"))
        sop_agent = SOPResearchAgent(sops_dir=sops_dir)
        rca_agent = RCAAnalysisAgent()
        intervention_agent = InterventionGovernanceAgent()

        state = {
            "line_id": "SMT-LINE-TEST",
            "defect_class": "short_circuit",
            "consecutive_count": 3,
        }

        # 1. SOP Research Agent
        res_sop = sop_agent.run(state)
        self.assertIn("sop_citations", res_sop)
        self.assertTrue(any("SOP-SMT-001" in c for c in res_sop["sop_citations"]))

        # 2. RCA Analysis Agent
        state.update(res_sop)
        res_rca = rca_agent.run(state)
        self.assertIn("rca_analysis", res_rca)
        self.assertIn("RCA", res_rca["rca_analysis"])

        # 3. Intervention Governance Agent
        state.update(res_rca)
        res_intervention = intervention_agent.run(state)
        self.assertEqual(res_intervention["proposed_action"], "HALT_LINE")
        self.assertTrue(res_intervention["requires_hitl"])

        # 4. Orchestrator alias works
        orchestrator = QualityIncidentOrchestrator(sops_dir=sops_dir)
        orch_res = orchestrator.run("short_circuit", 3)
        self.assertEqual(orch_res["proposed_action"], "HALT_LINE")

    def test_agent_hitl_interrupt_and_resume(self):
        """Verify that agent pauses at interrupt() and resumes with supervisor sign-off."""
        thread_id = "test-hitl-thread-42"
        paused_state = self.agent.run(
            defect_class="short_circuit",
            consecutive_count=3,
            line_id="SMT-LINE-01",
            thread_id=thread_id
        )
        self.assertEqual(paused_state["proposed_action"], "HALT_LINE")
        self.assertTrue(paused_state["requires_hitl"])
        self.assertIn("__interrupt__", paused_state)

        # Resume with Supervisor Approval
        resumed_state = self.agent.resume_approval(
            thread_id=thread_id,
            approved=True,
            supervisor_id="lead_qa_engineer"
        )
        self.assertEqual(resumed_state["approval_status"], "APPROVED")
        self.assertEqual(resumed_state["approved_by"], "lead_qa_engineer")
        self.assertIn("EXECUTED", resumed_state["execution_result"])

    def test_sop_search_ipc610_and_nozzle_maintenance(self):
        """BM25 search must correctly retrieve new IPC-A-610, reflow drift, and nozzle maintenance SOPs."""
        # 1. IPC Class 3
        res_ipc = self.retriever.search("IPC Class 3 solder fillet", top_k=1)
        self.assertGreater(len(res_ipc), 0)
        self.assertIn("SOP-SMT-004", res_ipc[0]["sop_id"])

        # 2. Reflow profile drift
        res_reflow = self.retriever.search("reflow profile drift TAL thermocouple", top_k=1)
        self.assertGreater(len(res_reflow), 0)
        self.assertIn("SOP-SMT-005", res_reflow[0]["sop_id"])

        # 3. Nozzle maintenance
        res_nozzle = self.retriever.search("vacuum nozzle ultrasonic cleaning", top_k=1)
        self.assertGreater(len(res_nozzle), 0)
        self.assertIn("SOP-SMT-006", res_nozzle[0]["sop_id"])

if __name__ == "__main__":
    unittest.main()

