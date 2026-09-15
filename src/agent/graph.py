import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from .rag import SOPRetriever
from .prompts import QUALITY_AGENT_SYSTEM_PROMPT

class AgentState(BaseModel):
    line_id: str = "SMT-LINE-01"
    defect_class: str
    consecutive_count: int = 3
    sop_citations: List[str] = []
    rca_analysis: str = ""
    proposed_action: str = ""
    requires_hitl: bool = True
    approval_status: str = "PENDING"

class QualityIncidentAgent:
    """
    LangGraph-compatible Smart Factory Incident Resolution Agent.
    Combines SOP Knowledge RAG, Groq LLM synthesis, and Human-in-the-Loop approval workflows.
    """

    def __init__(self, sops_dir: str = "data/sops"):
        self.retriever = SOPRetriever(sops_dir=sops_dir)
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    def run(self, defect_class: str, consecutive_count: int = 3, line_id: str = "SMT-LINE-01") -> Dict[str, Any]:
        """
        Executes the incident resolution pipeline:
        1. Query SOP RAG
        2. Synthesize RCA & Action
        3. Formulate HITL Action Ticket
        """
        # 1. RAG Search
        sop_results = self.retriever.search(defect_class, top_k=2)
        sop_context = "\n\n".join([f"[{r['sop_id']}]:\n{r['snippet']}" for r in sop_results])
        citations = [r['sop_id'] for r in sop_results]

        # 2. LLM Call or Deterministic Fallback
        rca_text = ""
        action = "HALT_LINE" if consecutive_count >= 3 and defect_class in ["short_circuit", "missing_component"] else "ROUTE_REWORK"

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
                print(f"[Agent] Groq API call note ({e}). Falling back to template.")
                rca_text = self._generate_fallback_rca(defect_class, consecutive_count, citations)
        else:
            rca_text = self._generate_fallback_rca(defect_class, consecutive_count, citations)

        return {
            "line_id": line_id,
            "defect_class": defect_class,
            "consecutive_count": consecutive_count,
            "sop_citations": citations,
            "rca_analysis": rca_text,
            "proposed_action": action,
            "requires_hitl": True,
            "approval_status": "PENDING"
        }

    def _generate_fallback_rca(self, defect_class: str, count: int, citations: List[str]) -> str:
        """Grounded, deterministic fallback response when offline or API key is unconfigured."""
        citation_str = ", ".join(citations) if citations else "SOP-SMT-001"
        if defect_class == "short_circuit":
            return (
                f"**Phân tích nguyên nhân gốc rễ (RCA)**: Phát hiện {count} lỗi hàn chập liên tiếp. "
                f"Hiện tượng này theo [{citation_str}] xuất phát từ việc tấm Stencil bị đọng cặn thiếc hàn dư thừa "
                f"hoặc nhiệt độ đỉnh của lò hàn hồi lưu vượt ngưỡng 260°C làm chảy loang kem hàn.\n\n"
                f"**Khuyến nghị can thiệp**: Bắt buộc tạm dừng dây chuyền (HALT_LINE) để kỹ sư bảo trì làm sạch tấm Stencil bằng cồn IPA và kiểm tra hồ sơ nhiệt độ.\n"
                f"**Trích dẫn quy trình**: [{citation_str}] Điều khoản 3."
            )
        elif defect_class == "missing_hole" or defect_class == "missing_component":
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
