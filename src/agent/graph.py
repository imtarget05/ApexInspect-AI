import os
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .rag import SOPRetriever
from .prompts import QUALITY_AGENT_SYSTEM_PROMPT

class AgentState(TypedDict):
    line_id: str
    defect_class: str
    consecutive_count: int
    sop_citations: List[str]
    sop_context: str
    rca_analysis: str
    proposed_action: str
    requires_hitl: bool
    approval_status: str

from .subagents import SOPResearchAgent, RCAAnalysisAgent, InterventionGovernanceAgent


class QualityIncidentAgent:
    """
    Production-grade LangGraph Multi-Agent Orchestrator for Smart Factory Incident Resolution.
    Orchestrates:
    - SOPResearchAgent: SOP manual RAG retrieval
    - RCAAnalysisAgent: Root cause analysis via Groq LLM / fallback
    - InterventionGovernanceAgent: Decision safety and Human-in-the-Loop governance.
    """

    def __init__(self, sops_dir: str = "data/sops"):
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model_name = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        self.sop_agent = SOPResearchAgent(sops_dir=sops_dir)
        self.rca_agent = RCAAnalysisAgent(groq_api_key=self.groq_api_key, model_name=self.model_name)
        self.intervention_agent = InterventionGovernanceAgent()
        self.retriever = self.sop_agent.retriever
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Constructs and compiles the LangGraph StateGraph workflow with decoupled sub-agents."""
        builder = StateGraph(AgentState)

        # 1. Add Workflow Nodes backed by specialist sub-agents
        builder.add_node("retrieve_sop", self.sop_agent.run)
        builder.add_node("synthesize_rca", self.rca_agent.run)
        builder.add_node("propose_action", self.intervention_agent.run)

        # 2. Add Workflow Edges
        builder.add_edge(START, "retrieve_sop")
        builder.add_edge("retrieve_sop", "synthesize_rca")
        builder.add_edge("synthesize_rca", "propose_action")
        builder.add_edge("propose_action", END)

        return builder.compile(checkpointer=self.checkpointer)

    def _node_retrieve_sop(self, state: AgentState) -> Dict[str, Any]:
        """Delegate to SOPResearchAgent for backward compatibility."""
        return self.sop_agent.run(state)

    def _node_synthesize_rca(self, state: AgentState) -> Dict[str, Any]:
        """Delegate to RCAAnalysisAgent for backward compatibility."""
        return self.rca_agent.run(state)

    def _node_propose_action(self, state: AgentState) -> Dict[str, Any]:
        """Delegate to InterventionGovernanceAgent for backward compatibility."""
        return self.intervention_agent.run(state)

    def _generate_fallback_rca(self, defect_class: str, count: int, citations: List[str]) -> str:
        """Delegate fallback to RCAAnalysisAgent."""
        return self.rca_agent._generate_fallback_rca(defect_class, count, citations)

    def run(self, defect_class: str, consecutive_count: int = 3, line_id: str = "SMT-LINE-01") -> Dict[str, Any]:
        """Invokes the compiled LangGraph StateGraph pipeline."""
        initial_state: AgentState = {
            "line_id": line_id,
            "defect_class": defect_class,
            "consecutive_count": consecutive_count,
            "sop_citations": [],
            "sop_context": "",
            "rca_analysis": "",
            "proposed_action": "",
            "requires_hitl": True,
            "approval_status": "PENDING"
        }
        config = {"configurable": {"thread_id": f"incident-{line_id}-{defect_class}"}}
        final_state = self.graph.invoke(initial_state, config=config)
        return final_state


QualityIncidentOrchestrator = QualityIncidentAgent
