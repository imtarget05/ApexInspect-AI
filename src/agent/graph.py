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

class QualityIncidentAgent:
    """
    Production-grade LangGraph StateGraph Agent for Smart Factory Incident Resolution.
    Orchestrates SOP RAG retrieval, Groq LLM synthesis, and Human-in-the-Loop decision governance.
    """

    def __init__(self, sops_dir: str = "data/sops"):
        self.retriever = SOPRetriever(sops_dir=sops_dir)
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model_name = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Constructs and compiles the LangGraph StateGraph workflow."""
        builder = StateGraph(AgentState)

        # 1. Add Workflow Nodes
        builder.add_node("retrieve_sop", self._node_retrieve_sop)
        builder.add_node("synthesize_rca", self._node_synthesize_rca)
        builder.add_node("propose_action", self._node_propose_action)

        # 2. Add Workflow Edges
        builder.add_edge(START, "retrieve_sop")
        builder.add_edge("retrieve_sop", "synthesize_rca")
        builder.add_edge("synthesize_rca", "propose_action")
        builder.add_edge("propose_action", END)

        return builder.compile(checkpointer=self.checkpointer)

    def _node_retrieve_sop(self, state: AgentState) -> Dict[str, Any]:
        """Node 1: Retrieves matching SOP documents from knowledge base."""
        defect_class = state["defect_class"]
        sop_results = self.retriever.search(defect_class, top_k=2)
        sop_context = "\n\n".join([f"[{r['sop_id']}]:\n{r['snippet']}" for r in sop_results])
        citations = [r['sop_id'] for r in sop_results]
        return {
            "sop_citations": citations,
            "sop_context": sop_context
        }

    def _node_synthesize_rca(self, state: AgentState) -> Dict[str, Any]:
        """Node 2: Generates Root Cause Analysis via Groq LLM or deterministic fallback."""
        defect_class = state["defect_class"]
        consecutive_count = state["consecutive_count"]
        line_id = state["line_id"]
        sop_context = state.get("sop_context", "")
        citations = state.get("sop_citations", [])

        rca_text = ""
        if self.groq_api_key and self.groq_api_key != "gsk_your_groq_api_key_here":
            try:
                from langchain_groq import ChatGroq
                from langchain_core.messages import SystemMessage, HumanMessage

                llm = ChatGroq(
                    api_key=self.groq_api_key,
                    model_name=self.model_name,
                    temperature=0.1
                )
                messages = [
                    SystemMessage(content=QUALITY_AGENT_SYSTEM_PROMPT),
                    HumanMessage(content=f"Sự cố phát hiện tại dây chuyền {line_id}:\n- Loại lỗi: {defect_class}\n- Số sản phẩm liên tiếp: {consecutive_count}\n\nTài liệu SOP liên quan trích xuất được:\n{sop_context}")
                ]
                response = llm.invoke(messages)
                rca_text = response.content
            except Exception as e:
                print(f"[Agent] Groq API call note ({e}). Using deterministic template.")
                rca_text = self._generate_fallback_rca(defect_class, consecutive_count, citations)
        else:
            rca_text = self._generate_fallback_rca(defect_class, consecutive_count, citations)

        return {"rca_analysis": rca_text}

    def _node_propose_action(self, state: AgentState) -> Dict[str, Any]:
        """Node 3: Formulates safe factory intervention action proposal."""
        defect_class = state["defect_class"]
        consecutive_count = state["consecutive_count"]

        action = "HALT_LINE" if consecutive_count >= 3 and defect_class in ["short_circuit", "missing_component", "short"] else "ROUTE_REWORK"
        return {
            "proposed_action": action,
            "requires_hitl": True,
            "approval_status": "PENDING"
        }

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

    def _generate_fallback_rca(self, defect_class: str, count: int, citations: List[str]) -> str:
        """Grounded, deterministic fallback response when offline or API key is unconfigured."""
        citation_str = ", ".join(citations) if citations else "SOP-SMT-001"
        if defect_class in ["short_circuit", "short"]:
            return (
                f"**Phân tích nguyên nhân gốc rễ (RCA)**: Phát hiện {count} lỗi hàn chập liên tiếp. "
                f"Hiện tượng này theo [{citation_str}] xuất phát từ việc tấm Stencil bị đọng cặn thiếc hàn dư thừa "
                f"hoặc nhiệt độ đỉnh của lò hàn hồi lưu vượt ngưỡng 260°C làm chảy loang kem hàn.\n\n"
                f"**Khuyến nghị can thiệp**: Bắt buộc tạm dừng dây chuyền (HALT_LINE) để kỹ sư bảo trì làm sạch tấm Stencil bằng cồn IPA và kiểm tra hồ sơ nhiệt độ.\n"
                f"**Trích dẫn quy trình**: [{citation_str}] Điều khoản 3."
            )
        elif defect_class in ["missing_hole", "missing_component"]:
            return (
                f"**Phân tích nguyên nhân gốc rễ (RCA)**: Phát hiện {count} lỗi thiếu linh kiện liên tiếp. "
                f"Theo [{citation_str}], nguyên nhân chủ yếu do đầu hút chân không (Vacuum Nozzle) của máy Pick & Place bị bám bụi bẩn hoặc cuộn nạp liệu Feeder bị kẹt.\n\n"
                f"**Khuyến nghị can thiệp**: Tạm dừng trạm cắm linh kiện để vệ sinh vòi hút và kiểm tra cảm biến nạp liệu.\n"
                f"**Trích dẫn quy trình**: [{citation_str}]."
            )
        else:
            return (
                f"**Phân tích nguyên nhân gốc rễ (RCA)**: Phát hiện lỗi ngoại quan '{defect_class}' bất thường. "
                f"Cần điều hướng các sản phẩm bị ảnh hưởng sang trạm Rework để kỹ thuật viên kiểm tra thủ công theo [{citation_str}]."
            )
