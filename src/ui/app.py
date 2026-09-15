import os
import sys
import time
import random
import cv2
import streamlit as st
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.vision.simulator import PCBCameraSimulator
from src.vision.detector import PCBDefectDetector
from src.agent.graph import QualityIncidentAgent
from src.agent.rag import SOPRetriever
from src.backend.database import init_db, SessionLocal
from src.backend.models import ProductionLine, InspectionLog, MESTicket

# Initialize Database on boot
init_db()

st.set_page_config(
    page_title="ApexInspect AI — Smart Factory Quality Platform",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Industrial High-Tech Look
st.markdown("""
<style>
    .metric-box {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 14px;
        border-left: 5px solid #00c853;
        margin-bottom: 10px;
    }
    .metric-box-alert {
        background-color: #2d1e1e;
        border-radius: 8px;
        padding: 14px;
        border-left: 5px solid #ff3d00;
        margin-bottom: 10px;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# App State Management
if "simulator" not in st.session_state:
    st.session_state.simulator = PCBCameraSimulator()
if "detector" not in st.session_state:
    st.session_state.detector = PCBDefectDetector()
if "agent" not in st.session_state:
    st.session_state.agent = QualityIncidentAgent()
if "retriever" not in st.session_state:
    st.session_state.retriever = SOPRetriever()

if "total_inspected" not in st.session_state:
    st.session_state.total_inspected = 0
if "defect_count" not in st.session_state:
    st.session_state.defect_count = 0
if "consecutive_defects" not in st.session_state:
    st.session_state.consecutive_defects = 0
if "line_status" not in st.session_state:
    st.session_state.line_status = "RUNNING"
if "active_incident" not in st.session_state:
    st.session_state.active_incident = None
if "defect_history" not in st.session_state:
    st.session_state.defect_history = []

# Sidebar Controls
with st.sidebar:
    st.title("🏭 ApexInspect AI")
    st.caption("Industrial Vision & Autonomous Quality Agent")
    st.divider()

    st.subheader("Dây chuyền sản xuất: SMT-LINE-01")
    status_color = "🟢" if st.session_state.line_status == "RUNNING" else "🔴"
    st.markdown(f"**Trạng thái hiện tại:** {status_color} `{st.session_state.line_status}`")

    if st.session_state.line_status == "HALTED":
        if st.button("🔄 Khởi Động Lại Dây Chuyền (Resume Line)", type="primary"):
            st.session_state.line_status = "RUNNING"
            st.session_state.consecutive_defects = 0
            st.session_state.active_incident = None
            st.success("Dây chuyền SMT-LINE-01 đã khởi động lại!")
            st.rerun()

    st.divider()
    st.subheader("🛠️ Công cụ Giả lập Dây chuyền")
    auto_defect_rate = st.slider("Tỷ lệ lỗi ngẫu nhiên (%)", min_value=0, max_value=50, value=10, step=5)
    
    st.markdown("**Tiêm lỗi thủ công (Test Trigger):**")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("⚡ Hàn chập (Short)"):
            st.session_state.force_defect = "short_circuit"
    with col_btn2:
        if st.button("🕳️ Mất lỗ (Hole)"):
            st.session_state.force_defect = "missing_hole"

    st.divider()
    st.markdown("⚙️ **Thông số Edge Inference:**")
    st.markdown("- Engine: `ONNX Runtime CPU`\n- Architecture: `YOLOv8 Nano`\n- Model Footprint: `14.2 MB`\n- Target Latency: `< 35 ms`")

# Header & Global KPI Metrics
st.title("Trung Tâm Điều Hành Chất Lượng & Thị Giác Máy Tính")
st.markdown("Giám sát lỗi lắp ráp linh kiện bề mặt thời gian thực & Điều phối sự cố thông minh qua AI Agent.")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
yield_rate = ((st.session_state.total_inspected - st.session_state.defect_count) / max(1, st.session_state.total_inspected)) * 100.0

with kpi_col1:
    st.metric("Tổng sản phẩm đã soi", f"{st.session_state.total_inspected:,}")
with kpi_col2:
    st.metric("Số lỗi phát hiện", f"{st.session_state.defect_count:,}", delta=f"{st.session_state.defect_count} vi phạm", delta_color="inverse")
with kpi_col3:
    st.metric("Tỷ lệ đạt chuẩn (Yield Rate)", f"{yield_rate:.1f}%", delta=f"{yield_rate - 95.0:.1f}% vs Mục tiêu 95%")
with kpi_col4:
    st.metric("Chuỗi lỗi liên tiếp", f"{st.session_state.consecutive_defects} / 3", delta="Cảnh báo dừng chuyền" if st.session_state.consecutive_defects >= 3 else "Bình thường")

st.divider()

# Main Dashboard Tabs
tab_vision, tab_agent, tab_analytics, tab_sops = st.tabs([
    "🎥 Live Line Vision (ONNX)", 
    "🚨 AI Incident & HITL Console", 
    "📊 Thống Kê & Pareto Lỗi", 
    "📚 Kho Quy Trình SOP"
])

# ==========================================
# TAB 1: LIVE VISION STREAM
# ==========================================
with tab_vision:
    col_stream, col_controls = st.columns([3, 1])

    with col_controls:
        st.subheader("Điều Khiển Camera")
        step_inspect = st.button("📸 Quét Sản Phẩm Tiếp Theo", type="primary", use_container_width=True)
        continuous_run = st.checkbox("Chế độ băng chuyền liên tục (Demo)", value=False)
        st.info("Mỗi lượt quét mô phỏng 1 bo mạch PCB vừa đi qua vùng camera soi quang học của máy SMT.")

    # Determine defect injection
    inject_defect = False
    chosen_defect = None

    if hasattr(st.session_state, "force_defect") and st.session_state.force_defect:
        inject_defect = True
        chosen_defect = st.session_state.force_defect
        st.session_state.force_defect = None
    elif random.randint(1, 100) <= auto_defect_rate:
        inject_defect = True

    if step_inspect or continuous_run or "last_frame" not in st.session_state:
        # Generate Synthetic Frame
        raw_frame, ground_truth = st.session_state.simulator.generate_pcb_frame(
            inject_defect=inject_defect,
            specific_defect=chosen_defect
        )
        # Infer with ONNX Detector
        annotated_frame, detections, latency_ms = st.session_state.detector.infer(raw_frame, ground_truth)
        st.session_state.last_frame = annotated_frame
        st.session_state.last_detections = detections
        st.session_state.last_latency = latency_ms

        # Update Telemetry & Metrics
        st.session_state.total_inspected += 1
        if len(detections) > 0:
            st.session_state.defect_count += 1
            st.session_state.consecutive_defects += 1
            def_type = detections[0].get("class", "defect")
            st.session_state.defect_history.append(def_type)

            # Check Incident Trigger (≥ 3 consecutive defects)
            if st.session_state.consecutive_defects >= 3 and not st.session_state.active_incident:
                # Trigger LangGraph Agent
                incident_result = st.session_state.agent.run(
                    defect_class=def_type,
                    consecutive_count=st.session_state.consecutive_defects,
                    line_id="SMT-LINE-01"
                )
                st.session_state.active_incident = incident_result
        else:
            st.session_state.consecutive_defects = 0

    with col_stream:
        if "last_frame" in st.session_state:
            # Convert BGR to RGB for Streamlit display
            rgb_frame = cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_frame, caption=f"Live Feed: SMT-LINE-01 Camera — Độ trễ suy luận: {st.session_state.last_latency:.1f}ms", use_container_width=True)

# ==========================================
# TAB 2: AI INCIDENT & HITL APPROVAL
# ==========================================
with tab_agent:
    st.subheader("Trung Tâm Xử Lý Sự Cố & Phê Duyệt Hành Động (Human-in-the-Loop)")

    if st.session_state.active_incident:
        inc = st.session_state.active_incident
        st.error(f"🚨 **SỰ CỐ KHẨN CẤP: PHÁT HIỆN {inc['consecutive_count']} LỖI '{inc['defect_class'].upper()}' LIÊN TIẾP**")
        
        col_inc1, col_inc2 = st.columns([2, 1])
        with col_inc1:
            st.markdown("### 🧠 Phân Tích Nguyên Nhân Gốc Rễ (RCA) từ AI Agent:")
            st.info(inc["rca_analysis"])
            st.markdown(f"**Tài liệu SOP liên quan:** `{', '.join(inc['sop_citations'])}`")
        
        with col_inc2:
            st.markdown("### ⚠️ Đề Xuất Hành Động:")
            st.warning(f"Lệnh đề xuất: **`{inc['proposed_action']}` (TẠM DỪNG DÂY CHUYỀN)**")
            st.caption("Yêu cầu chữ ký xác nhận của Quản đốc ca trước khi hệ thống MES thi hành lệnh dừng chuyền.")

            col_act1, col_act2 = st.columns(2)
            with col_act1:
                if st.button("✅ Phê Duyệt Dừng Chuyền", type="primary", use_container_width=True):
                    st.session_state.line_status = "HALTED"
                    st.session_state.active_incident = None
                    st.success("ĐÃ THI HÀNH: Dây chuyền SMT-LINE-01 đã được dừng khẩn cấp an toàn theo SOP-SMT-003!")
                    st.rerun()
            with col_act2:
                if st.button("❌ Bỏ Qua / Cảnh Báo Lại", use_container_width=True):
                    st.session_state.active_incident = None
                    st.session_state.consecutive_defects = 0
                    st.info("Đã hủy bỏ đề xuất. Dây chuyền tiếp tục vận hành.")
                    st.rerun()
    else:
        st.success("✅ **Hệ thống vận hành ổn định.** Chưa có sự cố lặp lại nào vượt ngưỡng cảnh báo.")

# ==========================================
# TAB 3: ANALYTICS & DEFECT PARETO
# ==========================================
with tab_analytics:
    st.subheader("Phân Tích Tỷ Lệ Lỗi (Pareto Chart)")
    if st.session_state.defect_history:
        from collections import Counter
        counts = Counter(st.session_state.defect_history)
        chart_data = {k: v for k, v in counts.items()}
        st.bar_chart(chart_data)
        st.caption("Biểu đồ phân bổ tần suất các loại lỗi phát hiện được trên dây chuyền.")
    else:
        st.info("Chưa có dữ liệu lỗi nào được ghi nhận. Hãy quét thêm sản phẩm ở Tab 1!")

# ==========================================
# TAB 4: SOP KNOWLEDGE BASE
# ==========================================
with tab_sops:
    st.subheader("Tra Cứu Quy Trình Vận Hành Chuẩn (SOP Knowledge Base)")
    search_q = st.text_input("Tìm kiếm tài liệu SOP:", placeholder="Ví dụ: short circuit, missing component, line halt...")
    
    if search_q:
        results = st.session_state.retriever.search(search_q, top_k=3)
        if results:
            for r in results:
                with st.expander(f"📄 {r['filename']} (Điểm khớp: {r['score']})"):
                    st.markdown(r['snippet'])
        else:
            st.warning("Không tìm thấy quy trình SOP nào phù hợp với từ khóa.")
    else:
        st.markdown("**Danh sách quy trình hiện có:**")
        for doc in st.session_state.retriever.documents:
            with st.expander(f"📄 {doc['filename']}"):
                st.markdown(doc['content'])
