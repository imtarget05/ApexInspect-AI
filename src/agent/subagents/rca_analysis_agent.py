"""RCA Analysis Sub-Agent – Analysis Agent synthesizing root cause analysis."""

import os
from typing import Dict, Any, List
from ..prompts import QUALITY_AGENT_SYSTEM_PROMPT


class RCAAnalysisAgent:
    """Specialist sub-agent for 5-Whys & Root Cause Analysis (RCA) synthesis."""

    def __init__(self, groq_api_key: str = "", model_name: str = "openai/gpt-oss-120b"):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "").strip()
        self.model_name = model_name or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
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
                    temperature=0.1,
                )
                messages = [
                    SystemMessage(content=QUALITY_AGENT_SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"Sự cố phát hiện tại dây chuyền {line_id}:\n"
                                f"- Loại lỗi: {defect_class}\n"
                                f"- Số sản phẩm liên tiếp: {consecutive_count}\n\n"
                                f"Tài liệu SOP liên quan trích xuất được:\n{sop_context}"
                    ),
                ]
                response = llm.invoke(messages)
                rca_text = response.content
            except Exception as e:
                print(f"[Agent] Groq API call note ({e}). Using deterministic template.")
                rca_text = self._generate_fallback_rca(defect_class, consecutive_count, citations)
        else:
            rca_text = self._generate_fallback_rca(defect_class, consecutive_count, citations)

        return {"rca_analysis": rca_text}

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
