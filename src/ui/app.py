import os
import sys
import uuid
import json
import datetime
import random
import cv2
import streamlit as st
import numpy as np
from collections import Counter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Synchronize Streamlit Secrets into os.environ (Streamlit Community Cloud compatibility)
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
except Exception:
    pass

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.vision.simulator import PCBCameraSimulator
from src.vision.detector import PCBDefectDetector
from src.agent.graph import QualityIncidentAgent
from src.agent.rag import SOPRetriever
from src.backend.database import init_db, seed_demo_data, SessionLocal
from src.backend.models import ProductionLine, InspectionLog, MESTicket, AuditLog
from src.backend.service import InspectionService
from src.backend.schemas import InspectionCreate

# Safely initialize Database and seed demo data
try:
    init_db()
    seed_demo_data(force=False)
except Exception as db_init_err:
    print(f"[App Startup Warning] Database init/seed: {db_init_err}")

st.set_page_config(
    page_title="ApexInspect AI — Smart Factory Quality Platform",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Industrial High-Tech Look (Enterprise SCADA / MES Dark Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre, .telemetry-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Modern Dashboard Header Banner */
    .scada-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    /* Pulse status badges */
    .status-pulse-running {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse-green 2s infinite;
        margin-right: 8px;
    }
    
    .status-pulse-halted {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #ef4444;
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
        animation: pulse-red 1.5s infinite;
        margin-right: 8px;
    }
    
    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    @keyframes pulse-red {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    /* Polished Metric & Incident Cards */
    div[data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600;
        font-size: 13px;
    }
    
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 24px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
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

    st.subheader("Dây chuyền: SMT-LINE-01")
    status_color = "🟢" if st.session_state.line_status == "RUNNING" else "🔴"
    st.markdown(f"**Trạng thái hiện tại:** {status_color} `{st.session_state.line_status}`")

    if st.session_state.line_status == "HALTED":
        if st.button("🔄 Khởi Động Lại Dây Chuyền (Resume)", type="primary"):
            db = SessionLocal()
            try:
                InspectionService.resume_line(db=db, line_id="SMT-LINE-01", operator_id="supervisor_ui", source_ip="streamlit_ui")
            finally:
                db.close()
            st.session_state.line_status = "RUNNING"
            st.session_state.consecutive_defects = 0
            st.session_state.active_incident = None
            st.success("Dây chuyền SMT-LINE-01 đã khởi động lại an toàn qua InspectionService!")
            st.rerun()

    st.divider()
    st.subheader("⚙️ Thông Số Kỹ Thuật Edge")
    st.markdown("- **Engine**: `ONNX Runtime CPU`")
    st.markdown("- **Architecture**: `YOLOv8 Nano`")
    st.markdown("- **Postprocessing**: `NMS (IoU 0.45)`")
    st.markdown("- **Latency Target**: `< 35 ms`")
    st.markdown(f"- **Agent LLM**: `{st.session_state.agent.model_name}`")
    st.markdown("- **Storage**: `Neon PostgreSQL`")

# Header & Global KPI Metrics (Industrial SCADA Topbar)
status_pulse_class = "status-pulse-running" if st.session_state.line_status == "RUNNING" else "status-pulse-halted"
status_text = "DÂY CHUYỀN HOẠT ĐỘNG BÌNH THƯỜNG" if st.session_state.line_status == "RUNNING" else "DÂY CHUYỀN ĐANG TẠM DỪNG (HALTED)"
status_text_color = "#10b981" if st.session_state.line_status == "RUNNING" else "#ef4444"

st.markdown(f"""
<div class="scada-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="{status_pulse_class}"></span>
                <span style="font-size: 11px; font-weight: 700; color: {status_text_color}; letter-spacing: 0.08em; font-family: 'JetBrains Mono', monospace;">
                    {status_text}
                </span>
            </div>
            <h1 style="margin: 6px 0 2px 0; font-size: 24px; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
                🏭 APEXINSPECT AI — TRUNG TÂM ĐIỀU HÀNH CHẤT LƯỢNG SMT
            </h1>
            <p style="margin: 0; color: #94a3b8; font-size: 13px;">
                Dây chuyền: <strong>SMT-LINE-01</strong> &bull; Engine: <strong>ONNX Runtime CPU (≈35.5 FPS)</strong> &bull; Incident Agent: <strong>LangGraph HITL Core</strong>
            </p>
        </div>
        <div style="text-align: right; background: rgba(15, 23, 42, 0.6); padding: 8px 14px; border-radius: 8px; border: 1px solid #334155;">
            <div style="font-size: 11px; color: #64748b; font-family: 'JetBrains Mono', monospace;">EDGE INFERENCE TARGET</div>
            <div style="font-size: 13px; font-weight: 700; color: #38bdf8; font-family: 'JetBrains Mono', monospace;">
                LATENCY &lt; 35ms &bull; FP32 GRAPH OPT
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Read historical metrics directly from Database
db = SessionLocal()
try:
    db_logs = db.query(InspectionLog).filter_by(line_id="SMT-LINE-01").order_by(InspectionLog.timestamp.desc()).all()
    display_total = len(db_logs)
    display_defects = sum(1 for log in db_logs if log.is_defective)

    # Compute current streak of consecutive defects from latest DB inspection records
    db_consecutive = 0
    for l in db_logs:
        if l.is_defective:
            db_consecutive += 1
        else:
            break

    # Sliding window for Yield Rate Drift (last 30 items)
    recent_30 = db_logs[:30]
    window_len = len(recent_30)
    window_defects = sum(1 for l in recent_30 if l.is_defective)
    window_error_rate = (window_defects / window_len * 100.0) if window_len > 0 else 0.0
    yield_drift_triggered = (window_len >= 10 and window_error_rate > 15.0)

    # Check pending tickets count
    pending_tickets_count = db.query(MESTicket).filter_by(status="PENDING_APPROVAL").count()
finally:
    db.close()

# Sync consecutive defects between DB and session
current_consecutive = max(st.session_state.get("consecutive_defects", 0), db_consecutive)
st.session_state.consecutive_defects = current_consecutive

yield_rate = ((display_total - display_defects) / max(1, display_total)) * 100.0

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.metric("Tổng sản phẩm đã soi (DB)", f"{display_total:,}")
with kpi_col2:
    st.metric("Số lỗi phát hiện (DB)", f"{display_defects:,}", delta=f"{display_defects} vi phạm", delta_color="inverse")
with kpi_col3:
    if yield_drift_triggered:
        st.metric(
            "Tỷ lệ đạt chuẩn (Yield Rate)", 
            f"{yield_rate:.1f}%", 
            delta=f"🚨 Trượt ngưỡng ({window_error_rate:.1f}% lỗi > 15%)", 
            delta_color="inverse"
        )
    else:
        st.metric(
            "Tỷ lệ đạt chuẩn (Yield Rate)", 
            f"{yield_rate:.1f}%", 
            delta=f"{yield_rate - 95.0:.1f}% vs Mục tiêu 95%"
        )
with kpi_col4:
    if current_consecutive >= 3:
        consecutive_delta = "🚨 Cảnh báo dừng chuyền"
    elif pending_tickets_count > 0:
        consecutive_delta = f"⚠️ {pending_tickets_count} sự cố chờ duyệt"
    else:
        consecutive_delta = "Bình thường"
    st.metric(
        "Chuỗi lỗi liên tiếp", 
        f"{current_consecutive} / 3", 
        delta=consecutive_delta, 
        delta_color="inverse" if current_consecutive >= 3 or pending_tickets_count > 0 else "normal"
    )

if yield_drift_triggered or pending_tickets_count > 0:
    st.warning(
        f"⚠️ **Cảnh báo chất lượng dây chuyền:** "
        f"{f'Tỷ lệ lỗi trượt ngưỡng ({window_error_rate:.1f}% > 15%). ' if yield_drift_triggered else ''}"
        f"{f'Có {pending_tickets_count} phiếu sự cố cần Quản đốc phê duyệt tại Tab 2 (AI Incident & HITL Console).' if pending_tickets_count > 0 else 'AI Agent đang giám sát điều phối.'}"
    )

st.divider()

# Main Dashboard Tabs
tab_vision, tab_agent, tab_analytics, tab_sops = st.tabs([
    "🎥 Soi Lỗi Băng Chuyền & Thử Nghiệm Ảnh", 
    "🚨 AI Incident & HITL Console", 
    "📊 Phân Tích Dữ Liệu & Lịch Sử MES", 
    "📚 Kho Quy Trình SOP & Thêm Mới"
])

# ==========================================
# TAB 1: LIVE VISION & MULTI-IMAGE TESTING
# ==========================================
with tab_vision:
    mode_choice = st.radio(
        "**Chọn Phương Thức Kiểm Tra Hình Ảnh:**",
        ["🤖 Băng Chuyền Tự Động (Stream Ảnh Mạch Thật)", "📁 Tải Ảnh Lên Từ Máy Tính (Upload Image)", "🖼️ Thư Viện Bo Mạch Mẫu (Sample Gallery)"],
        horizontal=True
    )

    col_stream, col_controls = st.columns([3, 1])

    # -------------------------------------------------------------
    # CASE 1: SIMULATOR STREAM
    # -------------------------------------------------------------
    if mode_choice == "🤖 Băng Chuyền Tự Động (Stream Ảnh Mạch Thật)":
        with col_controls:
            st.subheader("Điều Khiển Camera")
            st.caption("Luồng ảnh: Bo mạch quang học thực tế từ dataset công nghiệp.")
            step_inspect = st.button("📸 Quét Sản Phẩm Tiếp Theo", type="primary", use_container_width=True)
            continuous_run = st.checkbox("Chế độ băng chuyền liên tục", value=False)
            auto_defect_rate = st.slider("Tỷ lệ lỗi ngẫu nhiên (%)", min_value=0, max_value=50, value=15, step=5)
            
            st.markdown("**Tiêm lỗi nhanh:**")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("⚡ Hàn chập"):
                    st.session_state.force_defect = "short_circuit"
            with col_b2:
                if st.button("🕳️ Mất lỗ"):
                    st.session_state.force_defect = "missing_hole"

        inject_defect = False
        chosen_defect = None
        if hasattr(st.session_state, "force_defect") and st.session_state.force_defect:
            inject_defect = True
            chosen_defect = st.session_state.force_defect
            st.session_state.force_defect = None
        elif random.randint(1, 100) <= auto_defect_rate:
            inject_defect = True

        if step_inspect or continuous_run or "last_frame" not in st.session_state:
            raw_frame, ground_truth = st.session_state.simulator.generate_pcb_frame(
                inject_defect=inject_defect,
                specific_defect=chosen_defect
            )
            annotated_frame, detections, latency_ms = st.session_state.detector.infer(raw_frame, ground_truth)
            st.session_state.last_frame = annotated_frame
            st.session_state.last_detections = detections
            st.session_state.last_latency = latency_ms

            # Ingest telemetry & evaluate triggers via unified InspectionService
            is_def = len(detections) > 0
            if is_def:
                st.session_state.consecutive_defects += 1
            else:
                st.session_state.consecutive_defects = 0

            db = SessionLocal()
            try:
                payload = InspectionCreate(
                    line_id="SMT-LINE-01",
                    is_defective=is_def,
                    defect_classes=[d.get("class", "defect") for d in detections],
                    confidence_scores=[d.get("confidence", 0.9) for d in detections],
                    bounding_boxes=[d.get("bbox", []) for d in detections],
                    inference_time_ms=latency_ms
                )
                log_entry, incident_triggered, created_ticket_id = InspectionService.record_telemetry(
                    db=db,
                    payload=payload,
                    agent=st.session_state.agent
                )
                if incident_triggered and created_ticket_id:
                    tick = db.query(MESTicket).filter_by(ticket_id=created_ticket_id).first()
                    if tick:
                        st.session_state.active_incident = {
                            "ticket_id": tick.ticket_id,
                            "defect_class": payload.defect_classes[0] if payload.defect_classes else "defect",
                            "consecutive_count": st.session_state.consecutive_defects,
                            "rca_analysis": tick.root_cause_analysis,
                            "sop_citations": [tick.recommended_sop],
                            "proposed_action": tick.action_type
                        }
            except Exception as e:
                db.rollback()
                st.warning(f"Lỗi ghi nhận telemetry: {e}")
            finally:
                db.close()

        with col_stream:
            if "last_frame" in st.session_state:
                rgb_frame = cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_BGR2RGB)
                st.image(rgb_frame, caption=f"Live Camera SMT-LINE-01 — Độ trễ ONNX: {st.session_state.last_latency:.1f}ms", use_container_width=True)

    # -------------------------------------------------------------
    # CASE 2: UPLOAD CUSTOM IMAGE
    # -------------------------------------------------------------
    elif mode_choice == "📁 Tải Ảnh Lên Từ Máy Tính (Upload Image)":
        with col_controls:
            st.subheader("Tải Ảnh Lên")
            uploaded_file = st.file_uploader("Chọn file ảnh PCB (.jpg, .png)", type=["png", "jpg", "jpeg"])
            st.caption("Hệ thống sẽ nạp ảnh của bạn vào pipeline tiền xử lý OpenCV và suy luận bằng mô hình ONNX.")

        with col_stream:
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                custom_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if custom_img is not None:
                    annotated_frame, detections, latency_ms = st.session_state.detector.infer(custom_img)
                    rgb_res = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st.image(rgb_res, caption=f"Kết quả phân tích ảnh tải lên — Độ trễ suy luận: {latency_ms:.1f}ms", use_container_width=True)

                    if detections:
                        st.warning(f"⚠️ Phát hiện **{len(detections)} lỗi** trên ảnh tải lên: `{', '.join([d['class'] for d in detections])}`")
                        if st.button("🧠 Kích Hoạt Agent Phân Tích Lỗi Cho Ảnh Này", type="primary"):
                            primary_defect = detections[0]["class"]
                            now_utc = datetime.datetime.now(datetime.timezone.utc)
                            ticket_id = f"TICK-{now_utc.strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
                            thread_id = f"incident-SMT-LINE-01-{ticket_id}"
                            incident_res = st.session_state.agent.run(
                                defect_class=primary_defect,
                                consecutive_count=1,
                                line_id="SMT-LINE-01",
                                thread_id=thread_id
                            )
                            db = SessionLocal()
                            try:
                                db.add(MESTicket(
                                    ticket_id=ticket_id,
                                    line_id="SMT-LINE-01",
                                    severity="CRITICAL" if primary_defect in ["short_circuit", "short"] else "MEDIUM",
                                    trigger_reason=f"Kích hoạt phân tích sự cố cho lỗi '{primary_defect}' từ ảnh tải lên.",
                                    root_cause_analysis=incident_res.get("rca_analysis", ""),
                                    recommended_sop=", ".join(incident_res.get("sop_citations", [])) or "SOP-SMT-001",
                                    action_type=incident_res.get("proposed_action", "ROUTE_REWORK"),
                                    status="PENDING_APPROVAL",
                                    thread_id=thread_id,
                                    created_at=now_utc
                                ))
                                db.add(AuditLog(
                                    log_id=f"AUD-{uuid.uuid4().hex[:8].upper()}",
                                    operator_id="OPERATOR_UPLOAD",
                                    action="TRIGGER_CUSTOM_INSPECTION",
                                    line_id="SMT-LINE-01",
                                    ticket_id=ticket_id,
                                    details=f"Created custom inspection incident {ticket_id} for defect '{primary_defect}'.",
                                    source_ip="streamlit_ui"
                                ))
                                db.commit()
                                incident_res["ticket_id"] = ticket_id
                            finally:
                                db.close()
                            st.session_state.active_incident = incident_res
                            st.success(f"Đã tạo phiếu sự cố cho lỗi '{primary_defect}' và chuyển sang Tab 2 (AI Incident Center)!")
                    else:
                        st.success("✅ **Không phát hiện lỗi trên bo mạch (PASS)**.")
            else:
                st.info("👆 Vui lòng chọn một file ảnh từ máy tính ở khung bên phải để bắt đầu kiểm tra.")

    # -------------------------------------------------------------
    # CASE 3: SAMPLE GALLERY
    # -------------------------------------------------------------
    else:
        with col_controls:
            st.subheader("Chọn Mẫu Thử Nghiệm")
            sample_type = st.selectbox(
                "Danh sách mẫu bo mạch thực tế (Kaggle Dataset):",
                [
                    "1. Bo mạch chuẩn (PASS - Ảnh mạch quang học thật)",
                    "2. Bo mạch lỗi Hàn Chập (Short Circuit - Ảnh mạch thật)",
                    "3. Bo mạch lỗi Mất Lỗ Khoan (Missing Hole - Ảnh mạch thật)",
                    "4. Bo mạch lỗi Khuyết Mạch Đồng (Mouse Bite - Ảnh mạch thật)",
                    "5. Bo mạch lỗi Đứt Mạch (Open Circuit - Ảnh mạch thật)",
                    "6. Bo mạch lỗi Râu Đồng (Spur - Ảnh mạch thật)",
                    "7. Bo mạch lỗi Vết Đồng Dư (Spurious Copper - Ảnh mạch thật)"
                ]
            )
            st.caption("Nguồn: Ảnh chụp quang học kính hiển vi công nghiệp thực tế từ dataset `akhatova/pcb-defects`.")
            run_sample = st.button("🔍 Quét Mẫu Này", type="primary", use_container_width=True)

        defect_map = {
            "1. Bo mạch chuẩn (PASS - Ảnh mạch quang học thật)": (False, None),
            "2. Bo mạch lỗi Hàn Chập (Short Circuit - Ảnh mạch thật)": (True, "short_circuit"),
            "3. Bo mạch lỗi Mất Lỗ Khoan (Missing Hole - Ảnh mạch thật)": (True, "missing_hole"),
            "4. Bo mạch lỗi Khuyết Mạch Đồng (Mouse Bite - Ảnh mạch thật)": (True, "mouse_bite"),
            "5. Bo mạch lỗi Đứt Mạch (Open Circuit - Ảnh mạch thật)": (True, "open_circuit"),
            "6. Bo mạch lỗi Râu Đồng (Spur - Ảnh mạch thật)": (True, "spur"),
            "7. Bo mạch lỗi Vết Đồng Dư (Spurious Copper - Ảnh mạch thật)": (True, "spurious_copper")
        }

        with col_stream:
            is_inj, def_name = defect_map[sample_type]
            raw_s, gt_s = st.session_state.simulator.generate_pcb_frame(inject_defect=is_inj, specific_defect=def_name)
            ann_s, dets_s, lat_s = st.session_state.detector.infer(raw_s, gt_s)
            st.image(cv2.cvtColor(ann_s, cv2.COLOR_BGR2RGB), caption=f"Mẫu: {sample_type} | Độ trễ ONNX: {lat_s:.1f}ms", use_container_width=True)

            if dets_s:
                st.warning(f"Phát hiện lỗi: **`{dets_s[0]['class'].upper()}`** (Confidence: {dets_s[0]['confidence']*100:.1f}%)")
                if st.button("🧠 Kích Hoạt AI Agent Phân Tích & Tra Cứu SOP Cho Mẫu Này"):
                    now_utc = datetime.datetime.now(datetime.timezone.utc)
                    ticket_id = f"TICK-{now_utc.strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
                    thread_id = f"incident-SMT-LINE-01-{ticket_id}"
                    incident_res = st.session_state.agent.run(
                        defect_class=dets_s[0]["class"],
                        consecutive_count=3,
                        line_id="SMT-LINE-01",
                        thread_id=thread_id
                    )
                    db = SessionLocal()
                    try:
                        db.add(MESTicket(
                            ticket_id=ticket_id,
                            line_id="SMT-LINE-01",
                            severity="CRITICAL" if dets_s[0]["class"] in ["short_circuit", "short"] else "MEDIUM",
                            trigger_reason=f"Kích hoạt phân tích sự cố mẫu bo mạch lỗi '{dets_s[0]['class']}'.",
                            root_cause_analysis=incident_res.get("rca_analysis", ""),
                            recommended_sop=", ".join(incident_res.get("sop_citations", [])) or "SOP-SMT-001",
                            action_type=incident_res.get("proposed_action", "HALT_LINE"),
                            status="PENDING_APPROVAL",
                            thread_id=thread_id,
                            created_at=now_utc
                        ))
                        db.add(AuditLog(
                            log_id=f"AUD-{uuid.uuid4().hex[:8].upper()}",
                            operator_id="OPERATOR_GALLERY",
                            action="TRIGGER_GALLERY_INSPECTION",
                            line_id="SMT-LINE-01",
                            ticket_id=ticket_id,
                            details=f"Created gallery sample incident {ticket_id} for defect '{dets_s[0]['class']}'.",
                            source_ip="streamlit_ui"
                        ))
                        db.commit()
                        incident_res["ticket_id"] = ticket_id
                    finally:
                        db.close()
                    st.session_state.active_incident = incident_res
                    st.success("Đã tạo phiếu sự cố và chuyển thông tin sang Tab 2 (AI Incident Center)!")
            else:
                st.success("Bo mạch đạt chuẩn chất lượng xuất xưởng.")

# ==========================================
# TAB 2: AI INCIDENT & HITL APPROVAL
# ==========================================
with tab_agent:
    st.subheader("Trung Tâm Xử Lý Sự Cố & Phê Duyệt Hành Động (Human-in-the-Loop)")

    db = SessionLocal()
    try:
        pending_tickets = db.query(MESTicket)\
            .filter_by(status="PENDING_APPROVAL")\
            .order_by(MESTicket.created_at.desc())\
            .all()
    finally:
        db.close()

    if pending_tickets:
        st.error(f"🚨 **PHÁT HIỆN {len(pending_tickets)} PHIẾU SỰ CỐ CẦN QUẢN ĐỐC PHÊ DUYỆT (HUMAN-IN-THE-LOOP)**")

        for tick in pending_tickets:
            severity_icon = "🔴" if tick.severity == "CRITICAL" else "🟡"
            with st.container():
                st.markdown(f"### {severity_icon} Phiếu Sự Cố: `{tick.ticket_id}` — Mức độ: **{tick.severity}**")
                col_inc1, col_inc2 = st.columns([2, 1])
                with col_inc1:
                    st.markdown("#### 🧠 Phân Tích Nguyên Nhân Gốc Rễ (RCA) từ LangGraph Agent:")
                    st.info(tick.root_cause_analysis or "Đang phân tích...")
                    st.markdown(f"**Lý do kích hoạt:** {tick.trigger_reason}")
                    st.markdown(f"**Tài liệu SOP liên quan:** `{tick.recommended_sop}`")
                    st.caption(f"Dây chuyền: `{tick.line_id}` | Thời gian tạo: `{tick.created_at}`")

                with col_inc2:
                    st.markdown("#### ⚠️ Đề Xuất Hành Động:")
                    action_desc = "TẠM DỪNG DÂY CHUYỀN" if tick.action_type == "HALT_LINE" else "ĐIỀU HƯỚNG SANG TRẠM REWORK"
                    st.warning(f"Lệnh đề xuất: **`{tick.action_type}`** ({action_desc})")
                    st.caption("Yêu cầu chữ ký xác nhận của Quản đốc ca trước khi hệ thống MES thi hành lệnh.")

                    col_act1, col_act2 = st.columns(2)
                    with col_act1:
                        btn_approve_label = "✅ Phê Duyệt Dừng Chuyền" if tick.action_type == "HALT_LINE" else "✅ Phê Duyệt Hành Động"
                        if st.button(btn_approve_label, key=f"appr_{tick.ticket_id}", type="primary", use_container_width=True):
                            db = SessionLocal()
                            try:
                                res = InspectionService.resolve_ticket(
                                    db=db,
                                    ticket_id=tick.ticket_id,
                                    action="APPROVE",
                                    approved_by="supervisor_on_duty",
                                    agent=st.session_state.agent,
                                    source_ip="streamlit_ui"
                                )
                                if tick.action_type == "HALT_LINE":
                                    st.session_state.line_status = "HALTED"
                            finally:
                                db.close()
                            st.session_state.active_incident = None
                            st.success(f"ĐÃ THI HÀNH: {res.get('message', f'Phiếu {tick.ticket_id} đã được phê duyệt thành công!')}")
                            st.rerun()
                    with col_act2:
                        if st.button("❌ Từ Chối / Bỏ Qua", key=f"rej_{tick.ticket_id}", use_container_width=True):
                            db = SessionLocal()
                            try:
                                res = InspectionService.resolve_ticket(
                                    db=db,
                                    ticket_id=tick.ticket_id,
                                    action="REJECT",
                                    approved_by="supervisor_on_duty",
                                    agent=st.session_state.agent,
                                    source_ip="streamlit_ui"
                                )
                            finally:
                                db.close()
                            st.session_state.active_incident = None
                            st.session_state.consecutive_defects = 0
                            st.info(f"Đã từ chối đề xuất cho phiếu {tick.ticket_id}. Dây chuyền tiếp tục vận hành.")
                            st.rerun()
                st.divider()

    elif st.session_state.active_incident:
        inc = st.session_state.active_incident
        st.error(f"🚨 **SỰ CỐ KHẨN CẤP: PHÁT HIỆN {inc.get('consecutive_count', 1)} LỖI '{inc.get('defect_class', 'DEFECT').upper()}'**")
        col_inc1, col_inc2 = st.columns([2, 1])
        with col_inc1:
            st.markdown("### 🧠 Phân Tích Nguyên Nhân Gốc Rễ (RCA) từ LangGraph Agent:")
            st.info(inc.get("rca_analysis", "Đang phân tích..."))
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
                    db = SessionLocal()
                    try:
                        t_id = inc.get("ticket_id")
                        if t_id:
                            InspectionService.resolve_ticket(
                                db=db,
                                ticket_id=t_id,
                                action="APPROVE",
                                approved_by="supervisor_on_duty",
                                agent=st.session_state.agent,
                                source_ip="streamlit_ui"
                            )
                        else:
                            set_db_line_status("HALTED")
                    finally:
                        db.close()
                    st.session_state.line_status = "HALTED"
                    st.session_state.active_incident = None
                    st.success("ĐÃ THI HÀNH: Dây chuyền SMT-LINE-01 đã được dừng khẩn cấp an toàn theo SOP-SMT-003!")
                    st.rerun()
            with col_act2:
                if st.button("❌ Bỏ Qua / Cảnh Báo Lại", use_container_width=True):
                    t_id = inc.get("ticket_id")
                    if t_id:
                        db = SessionLocal()
                        try:
                            InspectionService.resolve_ticket(
                                db=db,
                                ticket_id=t_id,
                                action="REJECT",
                                approved_by="supervisor_on_duty",
                                agent=st.session_state.agent,
                                source_ip="streamlit_ui"
                            )
                        finally:
                            db.close()
                    st.session_state.active_incident = None
                    st.session_state.consecutive_defects = 0
                    st.info("Đã hủy bỏ đề xuất. Dây chuyền tiếp tục vận hành.")
                    st.rerun()
    else:
        st.success("✅ **Hệ thống vận hành ổn định.** Chưa có sự cố nào vượt ngưỡng cảnh báo cần phê duyệt.")

# ==========================================
# TAB 3: ANALYTICS & MES HISTORY
# ==========================================
with tab_analytics:
    col_t3_head, col_t3_btn = st.columns([3, 1])
    with col_t3_head:
        st.subheader("Phân Tích Dữ Liệu Kiểm Tra Từ Database (Neon / SQLite)")
    with col_t3_btn:
        if st.button("⚡ Nạp Thêm 50 Lô Kiểm Tra Mẫu", use_container_width=True):
            seed_demo_data(force=True)
            st.success("Đã nạp thêm 50 dữ liệu mẫu vào Database!")
            st.rerun()
    
    db = SessionLocal()
    try:
        defective_logs = db.query(InspectionLog).filter_by(is_defective=True).all()
        recent_tickets = db.query(MESTicket).order_by(MESTicket.created_at.desc()).limit(10).all()
        total_logs_count = db.query(InspectionLog).count()
    finally:
        db.close()

    col_an1, col_an2 = st.columns(2)
    with col_an1:
        st.markdown(f"#### 📊 Biểu Đồ Phân Bổ Loại Lỗi (Pareto Chart) — {len(defective_logs)} lỗi / {total_logs_count} mẫu")
        all_defects = []
        for l in defective_logs:
            classes = l.defect_classes
            if isinstance(classes, str):
                try:
                    classes = json.loads(classes)
                except Exception:
                    classes = [classes]
            if isinstance(classes, list):
                all_defects.extend(classes)

        if all_defects:
            counts = Counter(all_defects)
            st.bar_chart(dict(counts))
            st.caption("Thống kê số lượng vi phạm được bóc tách từ trường JSON của database.")
        else:
            st.info("Chưa có lỗi nào được ghi nhận trong database. Bấm nút 'Nạp Thêm 50 Lô Kiểm Tra Mẫu' ở trên để xem biểu đồ ngay!")

    with col_an2:
        st.markdown(f"#### 📋 Lịch Sử Phiếu Sự Cố MES Gần Đây ({len(recent_tickets)} phiếu)")
        if recent_tickets:
            for tick in recent_tickets:
                status_badge = "🟡 Chờ duyệt" if tick.status == "PENDING_APPROVAL" else ("🟢 Đã duyệt" if tick.status == "APPROVED" else "🔴 Từ chối")
                with st.expander(f"{tick.ticket_id} — {status_badge} ({tick.severity})"):
                    st.markdown(f"**Lý do:** {tick.trigger_reason}")
                    st.markdown(f"**SOP áp dụng:** `{tick.recommended_sop}`")
                    st.markdown(f"**Hành động:** `{tick.action_type}`")
                    st.markdown(f"**Phân tích nguyên nhân:** {tick.root_cause_analysis}")
                    if tick.approved_by:
                        st.caption(f"Người duyệt: {tick.approved_by} | Lúc: {tick.resolved_at}")
        else:
            st.info("Chưa có phiếu sự cố nào trong database. Hãy thử quét 3 lỗi liên tiếp ở Tab 1!")

# ==========================================
# TAB 4: SOP KNOWLEDGE BASE & NEW SOP CREATOR
# ==========================================
with tab_sops:
    st.subheader("Kho Tri Thức Quy Trình Vận Hành Chuẩn (SOP)")

    # Form Thêm Quy Trình Mới
    with st.expander("➕ **Thêm Quy Trình Vận Hành Chuẩn Mới (New SOP Manual)**", expanded=False):
        with st.form("new_sop_form"):
            st.markdown("#### Điền thông tin quy trình vận hành nhà máy mới:")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                new_sop_id = st.text_input("Mã tài liệu SOP:", value="SOP-SMT-004-stencil-wipe", help="Ví dụ: SOP-SMT-004-stencil-wipe")
                new_sop_area = st.text_input("Khu vực áp dụng:", value="Trạm in kem hàn & Trạm dán chip bề mặt")
            with col_f2:
                new_sop_title = st.text_input("Tiêu đề quy trình:", value="Quy trình lau tấm Stencil tự động và thủ công bằng cồn IPA")
                new_sop_risk = st.selectbox("Mức độ rủi ro:", ["CAO (CRITICAL)", "TRUNG BÌNH (MEDIUM)", "THẤP (LOW)"])

            new_sop_content = st.text_area(
                "Nội dung quy trình (Hỗ trợ định dạng Markdown):",
                height=180,
                value="""## 1. Định nghĩa & Dấu hiệu
Khi bo mạch xuất hiện lỗi kem hàn tràn hoặc hàn chập liên tục, tấm Stencil cần được vệ sinh ngay.

## 2. Nguyên nhân
Cặn thiếc hàn đọng ở mặt dưới tấm kim loại do cơ cấu gạt tự động hết cuộn giấy lau.

## 3. Quy trình khắc phục tiêu chuẩn
1. Tạm dừng chu trình in của máy.
2. Dùng giẻ sạch không bụi thấm cồn Isopropyl (IPA 99.7%) lau đều mặt dưới tấm Stencil.
3. Kiểm tra bằng mắt dưới ánh sáng đèn LED trước khi cho phép máy tiếp tục chạy.
"""
            )
            submit_sop = st.form_submit_button("💾 Lưu & Cập Nhật Vào Kho Tri Thức Agent", type="primary")

            if submit_sop:
                if new_sop_id and new_sop_content:
                    filename = f"{new_sop_id}.md"
                    sops_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/sops"))
                    filepath = os.path.join(sops_dir, filename)

                    full_doc = f"# {new_sop_id}: {new_sop_title}\n\n**Mã tài liệu**: {new_sop_id}  \n**Khu vực áp dụng**: {new_sop_area}  \n**Mức độ rủi ro**: {new_sop_risk}  \n\n---\n\n{new_sop_content}"
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(full_doc)

                    # Reload retriever
                    st.session_state.retriever._load_documents()
                    st.success(f"🎉 Đã lưu thành công quy trình `{filename}`! AI Agent hiện tại đã có thể tra cứu tài liệu này.")
                    st.rerun()
                else:
                    st.error("Vui lòng nhập đầy đủ Mã tài liệu và Nội dung quy trình.")

    st.divider()

    # Search SOP
    search_q = st.text_input("🔍 Tìm kiếm tài liệu SOP:", placeholder="Ví dụ: short circuit, missing component, stencil, line halt...")
    if search_q:
        results = st.session_state.retriever.search(search_q, top_k=4)
        if results:
            for r in results:
                with st.expander(f"📄 {r['filename']} (Điểm khớp: {r['score']})", expanded=True):
                    st.markdown(r['snippet'])
        else:
            st.warning("Không tìm thấy quy trình SOP nào phù hợp với từ khóa.")
    else:
        st.markdown(f"**Danh sách quy trình hiện có ({len(st.session_state.retriever.documents)} tài liệu):**")
        for doc in st.session_state.retriever.documents:
            with st.expander(f"📄 {doc['filename']}"):
                st.markdown(doc['content'])
