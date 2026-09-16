import os
import uuid
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from .rag import SOPRetriever
from .prompts import QUALITY_AGENT_SYSTEM_PROMPT
from .subagents import SOPResearchAgent, RCAAnalysisAgent, InterventionGovernanceAgent


class AgentState(TypedDict, total=False):
    line_id: str
    defect_class: str
    consecutive_count: int
    sop_citations: List[str]
    sop_context: str
    rca_analysis: str
    proposed_action: str
    requires_hitl: bool
    approval_status: str
    approved_by: Optional[str]
    thread_id: Optional[str]
    execution_result: Optional[str]


class QualityIncidentAgent:
    """
    Production-grade LangGraph Multi-Agent Orchestrator for Smart Factory Incident Resolution.
    Orchestrates:
    - SOPResearchAgent: SOP manual RAG retrieval
    - RCAAnalysisAgent: Root cause analysis via Groq LLM / fallback
    - InterventionGovernanceAgent: Decision safety and action proposal
    - await_supervisor_approval: True LangGraph interrupt() checkpoint
    - execute_factory_action: Post-approval execution node
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
        """Constructs and compiles the LangGraph StateGraph workflow with true interrupt() HITL."""
        builder = StateGraph(AgentState)

        # 1. Add Workflow Nodes
        builder.add_node("retrieve_sop", self.sop_agent.run)
        builder.add_node("synthesize_rca", self.rca_agent.run)
        builder.add_node("propose_action", self.intervention_agent.run)
        builder.add_node("await_supervisor_approval", self._node_await_supervisor_approval)
        builder.add_node("execute_factory_action", self._node_execute_factory_action)

        # 2. Add Workflow Edges
        builder.add_edge(START, "retrieve_sop")
        builder.add_edge("retrieve_sop", "synthesize_rca")
        builder.add_edge("synthesize_rca", "propose_action")
        builder.add_edge("propose_action", "await_supervisor_approval")
        builder.add_edge("await_supervisor_approval", "execute_factory_action")
        builder.add_edge("execute_factory_action", END)

        return builder.compile(checkpointer=self.checkpointer)

    def _node_await_supervisor_approval(self, state: AgentState) -> Dict[str, Any]:
        """Halts the StateGraph using LangGraph's native interrupt() primitive until a supervisor signs off."""
        if state.get("requires_hitl", True):
            decision = interrupt({
                "action": state.get("proposed_action", "HALT_LINE"),
                "line_id": state.get("line_id", "SMT-LINE-01"),
                "defect_class": state.get("defect_class", "defect"),
                "consecutive_count": state.get("consecutive_count", 1),
                "rca_summary": state.get("rca_analysis", "")[:300]
            })
            approved = decision.get("approved", False) if isinstance(decision, dict) else bool(decision)
            supervisor = decision.get("supervisor_id", "supervisor_on_duty") if isinstance(decision, dict) else "supervisor_on_duty"
            return {
                "approval_status": "APPROVED" if approved else "REJECTED",
                "approved_by": supervisor,
                "requires_hitl": False
            }
        return {"approval_status": "AUTO_APPROVED", "requires_hitl": False}

    def _node_execute_factory_action(self, state: AgentState) -> Dict[str, Any]:
        """Executes the approved action upon supervisor resume."""
        action = state.get("proposed_action", "HALT_LINE")
        status = state.get("approval_status", "PENDING")
        line_id = state.get("line_id", "SMT-LINE-01")
        supervisor = state.get("approved_by", "system")

        if status == "APPROVED":
            result = f"EXECUTED: Action {action} successfully dispatched to line {line_id}."
        else:
            result = f"DISMISSED: Action {action} rejected by {supervisor}. Line {line_id} remains operational."

        return {"execution_result": result}

    def _node_retrieve_sop(self, state: AgentState) -> Dict[str, Any]:
        return self.sop_agent.run(state)

    def _node_synthesize_rca(self, state: AgentState) -> Dict[str, Any]:
        return self.rca_agent.run(state)

    def _node_propose_action(self, state: AgentState) -> Dict[str, Any]:
        return self.intervention_agent.run(state)

    def _generate_fallback_rca(self, defect_class: str, count: int, citations: List[str]) -> str:
        return self.rca_agent._generate_fallback_rca(defect_class, count, citations)

    def run(self, defect_class: str, consecutive_count: int = 3, line_id: str = "SMT-LINE-01", thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Invokes the LangGraph StateGraph pipeline.
        Stops at the await_supervisor_approval node via interrupt() when requires_hitl is True.
        """
        if not thread_id:
            thread_id = f"incident-{line_id}-{defect_class}-{uuid.uuid4().hex[:6]}"

        initial_state: AgentState = {
            "line_id": line_id,
            "defect_class": defect_class,
            "consecutive_count": consecutive_count,
            "sop_citations": [],
            "sop_context": "",
            "rca_analysis": "",
            "proposed_action": "",
            "requires_hitl": True,
            "approval_status": "PENDING",
            "thread_id": thread_id
        }
        config = {"configurable": {"thread_id": thread_id}}
        state_result = self.graph.invoke(initial_state, config=config)
        state_result["thread_id"] = thread_id
        return state_result

    def resume_approval(self, thread_id: str, approved: bool, supervisor_id: str = "supervisor_on_duty") -> Dict[str, Any]:
        """
        Resumes the interrupted StateGraph workflow with the human supervisor's decision.
        """
        config = {"configurable": {"thread_id": thread_id}}
        resumed_state = self.graph.invoke(
            Command(resume={"approved": approved, "supervisor_id": supervisor_id}),
            config=config
        )
        resumed_state["thread_id"] = thread_id
        return resumed_state


QualityIncidentOrchestrator = QualityIncidentAgent
