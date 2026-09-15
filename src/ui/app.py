import os
import sys
import uuid
import datetime
import random
import cv2
import streamlit as st
import numpy as np
from collections import Counter

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

# Helper function to get current DB line status
def get_db_line_status(line_id: str = "SMT-LINE-01") -> str:
    db = SessionLocal()
    try:
        line = db.query(ProductionLine).filter_by(line_id=line_id).first()
        return line.status if line else "RUNNING"
    finally:
        db.close()

# Helper function to set DB line status
def set_db_line_status(status: str, line_id: str = "SMT-LINE-01"):
    db = SessionLocal()
    try:
        line = db.query(ProductionLine).filter_by(line_id=line_id).first()
        if line:
            line.status = status
            db.commit()
    finally:
        db.close()

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
    st.session_state.line_status = get_db_line_status()
if "active_incident" not in st.session_state:
    st.session_state.active_incident = None

# Sync line status with DB
st.session_state.line_status = get_db_line_status()

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
            set_db_line_status("RUNNING")
            st.session_state.line_status = "RUNNING"
            st.session_state.consecutive_defects = 0
            st.session_state.active_incident = None
            st.success("Dây chuyền SMT-LINE-01 đã khởi động lại an toàn!")
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
    st.markdown("- Engine: `ONNX Runtime CPU`\n- Architecture: `YOLOv8 Nano`\n- Classes: `6 PCB Defect Classes`\n- Target Latency: `< 35 ms`")

# Header & Global KPI Metrics
st.title("Trung Tâm Điều Hành Chất Lượng & Thị Giác Máy Tính")
st.markdown("Giám sát lỗi lắp ráp linh kiện bề mặt thời gian thực & Điều phối sự cố thông minh qua AI Agent.")

# Read historical totals from DB
db = SessionLocal()
try:
    db_logs = db.query(InspectionLog).filter_by(line_id="SMT-LINE-01").all()
    db_total = len(db_logs)
    db_defects = sum(1 for log in db_logs if log.is_defective)
finally:
    db.close()

display_total = max(db_total, st.session_state.total_inspected)
display_defects = max(db_defects, st.session_state.defect_count)
yield_rate = ((display_total - display_defects) / max(1, display_total)) * 100.0

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.metric("Tổng sản phẩm đã soi", f"{display_total:,}")
with kpi_col2:
    st.metric("Số lỗi phát hiện", f"{display_defects:,}", delta=f"{display_defects} vi phạm", delta_color="inverse")
with kpi_col3:
    st.metric("Tỷ lệ đạt chuẩn (Yield Rate)", f"{yield_rate:.1f}%", delta=f"{yield_rate - 95.0:.1f}% vs Mục tiêu 95%")
with kpi_col4:
    st.metric("Chuỗi lỗi liên tiếp", f"{st.session_state.consecutive_defects} / 3", delta="Cảnh báo dừng chuyền" if st.session_state.consecutive_defects >= 3 else "Bình thường")

st.divider()

# Main Dashboard Tabs
tab_vision, tab_agent, tab_analytics, tab_sops = st.tabs([
    "🎥 Live Line Vision (ONNX)", 
    "🚨 AI Incident & HITL Console", 
    "📊 Thống Kê & Lịch Sử MES", 
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

    inject_defect = False
    chosen_defect = None

    if hasattr(st.session_state, "force_defect") and st.session_state.force_defect:
        inject_defect = True
        chosen_defect = st.session_state.force_defect
        st.session_state.force_defect = None
    elif random.randint(1, 100) <= auto_defect_rate:
        inject_defect = True

    if step_inspect or continuous_run or "last_frame" not in st.session_state:
        # 1. Generate Frame
        raw_frame, ground_truth = st.session_state.simulator.generate_pcb_frame(
            inject_defect=inject_defect,
            specific_defect=chosen_defect
        )
        # 2. Run ONNX Inference
        annotated_frame, detections, latency_ms = st.session_state.detector.infer(raw_frame, ground_truth)
        st.session_state.last_frame = annotated_frame
        st.session_state.last_detections = detections
        st.session_state.last_latency = latency_ms

        # 3. Persist Inspection Telemetry to Database
        st.session_state.total_inspected += 1
        is_def = len(detections) > 0
        def_classes = [d.get("class", "defect") for d in detections]
        conf_scores = [d.get("confidence", 0.9) for d in detections]
        bboxes = [d.get("bbox", []) for d in detections]

        db = SessionLocal()
        try:
            insp_id = f"INSP-{uuid.uuid4().hex[:8].upper()}"
            log_entry = InspectionLog(
                inspection_id=insp_id,
                line_id="SMT-LINE-01",
                is_defective=is_def,
                defect_classes=def_classes,
                confidence_scores=conf_scores,
                bounding_boxes=bboxes,
                inference_time_ms=latency_ms
            )
            db.add(log_entry)
            db.commit()
        except Exception as e:
            print(f"[UI] Database write error: {e}")
            db.rollback()
        finally:
            db.close()

        # 4. Check Consecutive Defect Trigger
        if is_def:
            st.session_state.defect_count += 1
            st.session_state.consecutive_defects += 1
            def_type = def_classes[0]

            if st.session_state.consecutive_defects >= 3 and not st.session_state.active_incident:
                # Trigger LangGraph Incident Resolution Agent
                incident_result = st.session_state.agent.run(
                    defect_class=def_type,
                    consecutive_count=st.session_state.consecutive_defects,
                    line_id="SMT-LINE-01"
                )
                
                # Persist MESTicket to Database
                db = SessionLocal()
                try:
                    now_utc = datetime.datetime.now(datetime.timezone.utc)
                    ticket_id = f"TICK-{now_utc.strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
                    ticket = MESTicket(
                        ticket_id=ticket_id,
                        line_id="SMT-LINE-01",
                        severity="CRITICAL" if def_type in ["short_circuit", "short"] else "MEDIUM",
                        trigger_reason=f"Phát hiện 3 sản phẩm liên tiếp có lỗi '{def_type}' trên chuyền SMT-LINE-01.",
                        root_cause_analysis=incident_result.get("rca_analysis", ""),
                        recommended_sop=", ".join(incident_result.get("sop_citations", [])) or "SOP-SMT-001",
                        action_type=incident_result.get("proposed_action", "HALT_LINE"),
                        status="PENDING_APPROVAL",
                        created_at=now_utc
                    )
                    db.add(ticket)
                    db.commit()
                    incident_result["ticket_id"] = ticket_id
                except Exception as e:
                    print(f"[UI] Error creating ticket: {e}")
                    db.rollback()
                finally:
                    db.close()

                st.session_state.active_incident = incident_result
        else:
            st.session_state.consecutive_defects = 0

    with col_stream:
        if "last_frame" in st.session_state:
            rgb_frame = cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_frame, caption=f"Live Feed: SMT-LINE-01 Camera — Độ trễ ONNX: {st.session_state.last_latency:.1f}ms", use_container_width=True)

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
            st.markdown("### 🧠 Phân Tích Nguyên Nhân Gốc Rễ (RCA) từ LangGraph Agent:")
            st.info(inc["rca_analysis"])
            st.markdown(f"**Tài liệu SOP liên quan:** `{', '.join(inc.get('sop_citations', []))}`")
            if "ticket_id" in inc:
                st.caption(f"Mã phiếu MES: `{inc['ticket_id']}`")
        
        with col_inc2:
            st.markdown("### ⚠️ Đề Xuất Hành Động:")
            st.warning(f"Lệnh đề xuất: **`{inc.get('proposed_action', 'HALT_LINE')}` (TẠM DỪNG DÂY CHUYỀN)**")
            st.caption("Yêu cầu chữ ký xác nhận của Quản đốc ca trước khi hệ thống MES thi hành lệnh dừng chuyền.")

            col_act1, col_act2 = st.columns(2)
            with col_act1:
                if st.button("✅ Phê Duyệt Dừng Chuyền", type="primary", use_container_width=True):
                    # Persist approval to DB
                    set_db_line_status("HALTED")
                    if "ticket_id" in inc:
                        db = SessionLocal()
                        try:
                            t = db.query(MESTicket).filter_by(ticket_id=inc["ticket_id"]).first()
                            if t:
                                t.status = "APPROVED"
                                t.approved_by = "supervisor_on_duty"
                                t.resolved_at = datetime.datetime.now(datetime.timezone.utc)
                                db.commit()
                        finally:
                            db.close()

                    st.session_state.line_status = "HALTED"
                    st.session_state.active_incident = None
                    st.success("ĐÃ THI HÀNH: Dây chuyền SMT-LINE-01 đã được dừng khẩn cấp an toàn theo SOP-SMT-003!")
                    st.rerun()
            with col_act2:
                if st.button("❌ Bỏ Qua / Cảnh Báo Lại", use_container_width=True):
                    if "ticket_id" in inc:
                        db = SessionLocal()
                        try:
                            t = db.query(MESTicket).filter_by(ticket_id=inc["ticket_id"]).first()
                            if t:
                                t.status = "REJECTED"
                                t.approved_by = "supervisor_on_duty"
                                t.resolved_at = datetime.datetime.now(datetime.timezone.utc)
                                db.commit()
                        finally:
                            db.close()

                    st.session_state.active_incident = None
                    st.session_state.consecutive_defects = 0
                    st.info("Đã hủy bỏ đề xuất. Dây chuyền tiếp tục vận hành.")
                    st.rerun()
    else:
        st.success("✅ **Hệ thống vận hành ổn định.** Chưa có sự cố lặp lại nào vượt ngưỡng cảnh báo.")

# ==========================================
# TAB 3: ANALYTICS & MES HISTORY
# ==========================================
with tab_analytics:
    st.subheader("Phân Tích Dữ Liệu Kiểm Tra Từ Database")
    
    db = SessionLocal()
    try:
        defective_logs = db.query(InspectionLog).filter_by(is_defective=True).all()
        recent_tickets = db.query(MESTicket).order_by(MESTicket.created_at.desc()).limit(5).all()
    finally:
        db.close()

    col_an1, col_an2 = st.columns(2)
    with col_an1:
        st.markdown("#### 📊 Biểu Đồ Phân Bổ Loại Lỗi (Pareto Chart)")
        all_defects = []
        for l in defective_logs:
            if l.defect_classes:
                all_defects.extend(l.defect_classes)

        if all_defects:
            counts = Counter(all_defects)
            st.bar_chart(dict(counts))
        else:
            st.info("Chưa có lỗi nào được ghi nhận trong database.")

    with col_an2:
        st.markdown("#### 📋 Lịch Sử Phiếu Sự Cố MES Gần Đây")
        if recent_tickets:
            for tick in recent_tickets:
                status_badge = "🟡 Chờ duyệt" if tick.status == "PENDING_APPROVAL" else ("🟢 Đã duyệt" if tick.status == "APPROVED" else "🔴 Từ chối")
                with st.expander(f"{tick.ticket_id} — {status_badge} ({tick.severity})"):
                    st.markdown(f"**Lý do:** {tick.trigger_reason}")
                    st.markdown(f"**SOP áp dụng:** `{tick.recommended_sop}`")
                    st.markdown(f"**Hành động:** `{tick.action_type}`")
        else:
            st.info("Chưa có phiếu sự cố nào trong database.")

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
