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

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.vision.simulator import PCBCameraSimulator
from src.vision.detector import PCBDefectDetector
from src.agent.graph import QualityIncidentAgent
from src.agent.rag import SOPRetriever
from src.backend.database import init_db, seed_demo_data, SessionLocal
from src.backend.models import ProductionLine, InspectionLog, MESTicket

# Initialize Database and seed demo data if empty
init_db()
seed_demo_data(force=False)

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

    st.subheader("Dây chuyền: SMT-LINE-01")
    status_color = "🟢" if st.session_state.line_status == "RUNNING" else "🔴"
    st.markdown(f"**Trạng thái hiện tại:** {status_color} `{st.session_state.line_status}`")

    if st.session_state.line_status == "HALTED":
        if st.button("🔄 Khởi Động Lại Dây Chuyền (Resume)", type="primary"):
            set_db_line_status("RUNNING")
            st.session_state.line_status = "RUNNING"
            st.session_state.consecutive_defects = 0
            st.session_state.active_incident = None
            st.success("Dây chuyền SMT-LINE-01 đã khởi động lại an toàn!")
            st.rerun()

    st.divider()
    st.subheader("⚙️ Thông Số Kỹ Thuật Edge")
    st.markdown("- **Engine**: `ONNX Runtime CPU`")
    st.markdown("- **Architecture**: `YOLOv8 Nano`")
    st.markdown("- **Postprocessing**: `NMS (IoU 0.45)`")
    st.markdown("- **Latency Target**: `< 35 ms`")
    st.markdown(f"- **Agent LLM**: `{st.session_state.agent.model_name}`")
    st.markdown("- **Storage**: `Neon PostgreSQL`")

# Header & Global KPI Metrics
st.title("Trung Tâm Điều Hành Chất Lượng & Thị Giác Máy Tính")
st.markdown("Giám sát lỗi lắp ráp linh kiện bề mặt thời gian thực & Điều phối sự cố thông minh qua AI Agent.")

# Read historical metrics directly from Database
db = SessionLocal()
try:
    db_logs = db.query(InspectionLog).filter_by(line_id="SMT-LINE-01").all()
    display_total = len(db_logs)
    display_defects = sum(1 for log in db_logs if log.is_defective)
finally:
    db.close()

yield_rate = ((display_total - display_defects) / max(1, display_total)) * 100.0

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.metric("Tổng sản phẩm đã soi (DB)", f"{display_total:,}")
with kpi_col2:
    st.metric("Số lỗi phát hiện (DB)", f"{display_defects:,}", delta=f"{display_defects} vi phạm", delta_color="inverse")
with kpi_col3:
    st.metric("Tỷ lệ đạt chuẩn (Yield Rate)", f"{yield_rate:.1f}%", delta=f"{yield_rate - 95.0:.1f}% vs Mục tiêu 95%")
with kpi_col4:
    st.metric("Chuỗi lỗi liên tiếp", f"{st.session_state.consecutive_defects} / 3", delta="Cảnh báo dừng chuyền" if st.session_state.consecutive_defects >= 3 else "Bình thường")

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
        ["🤖 Băng Chuyền Tự Động (Simulator Stream)", "📁 Tải Ảnh Lên Từ Máy Tính (Upload Image)", "🖼️ Thư Viện Bo Mạch Mẫu (Sample Gallery)"],
        horizontal=True
    )

    col_stream, col_controls = st.columns([3, 1])

    # -------------------------------------------------------------
    # CASE 1: SIMULATOR STREAM
    # -------------------------------------------------------------
    if mode_choice == "🤖 Băng Chuyền Tự Động (Simulator Stream)":
        with col_controls:
            st.subheader("Điều Khiển Camera")
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

            # Log to DB
            is_def = len(detections) > 0
            def_classes = [d.get("class", "defect") for d in detections]
            conf_scores = [d.get("confidence", 0.9) for d in detections]
            bboxes = [d.get("bbox", []) for d in detections]

            db = SessionLocal()
            try:
                insp_id = f"INSP-{uuid.uuid4().hex[:8].upper()}"
                db.add(InspectionLog(
                    inspection_id=insp_id,
                    line_id="SMT-LINE-01",
                    is_defective=is_def,
                    defect_classes=def_classes,
                    confidence_scores=conf_scores,
                    bounding_boxes=bboxes,
                    inference_time_ms=latency_ms
                ))
                db.commit()
            except Exception as e:
                db.rollback()
            finally:
                db.close()

            # Consecutive defects check
            if is_def:
                st.session_state.consecutive_defects += 1
                def_type = def_classes[0]
                if st.session_state.consecutive_defects >= 3 and not st.session_state.active_incident:
                    incident_result = st.session_state.agent.run(
                        defect_class=def_type,
                        consecutive_count=st.session_state.consecutive_defects,
                        line_id="SMT-LINE-01"
                    )
                    # Persist ticket
                    db = SessionLocal()
                    try:
                        now_utc = datetime.datetime.now(datetime.timezone.utc)
                        ticket_id = f"TICK-{now_utc.strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
                        db.add(MESTicket(
                            ticket_id=ticket_id,
                            line_id="SMT-LINE-01",
                            severity="CRITICAL" if def_type in ["short_circuit", "short"] else "MEDIUM",
                            trigger_reason=f"Phát hiện 3 sản phẩm liên tiếp có lỗi '{def_type}' trên chuyền SMT-LINE-01.",
                            root_cause_analysis=incident_result.get("rca_analysis", ""),
                            recommended_sop=", ".join(incident_result.get("sop_citations", [])) or "SOP-SMT-001",
                            action_type=incident_result.get("proposed_action", "HALT_LINE"),
                            status="PENDING_APPROVAL",
                            created_at=now_utc
                        ))
                        db.commit()
                        incident_result["ticket_id"] = ticket_id
                    finally:
                        db.close()
                    st.session_state.active_incident = incident_result
            else:
                st.session_state.consecutive_defects = 0

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
                            incident_res = st.session_state.agent.run(
                                defect_class=primary_defect,
                                consecutive_count=1,
                                line_id="SMT-LINE-01"
                            )
                            st.session_state.active_incident = incident_res
                            st.success(f"Đã chuyển sự cố lỗi '{primary_defect}' sang Tab 2 (AI Incident Center)!")
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
                "Danh sách mẫu bo mạch có sẵn:",
                [
                    "1. Bo mạch chuẩn (PASS - Không lỗi)",
                    "2. Bo mạch lỗi Hàn Chập (Short Circuit)",
                    "3. Bo mạch lỗi Mất Lỗ Khoan (Missing Hole)",
                    "4. Bo mạch lỗi Khuyết Mạch Đồng (Mouse Bite)",
                    "5. Bo mạch lỗi Đứt Mạch (Open Circuit)",
                    "6. Bo mạch lỗi Râu Đồng (Spur)"
                ]
            )
            run_sample = st.button("🔍 Quét Mẫu Này", type="primary", use_container_width=True)

        defect_map = {
            "1. Bo mạch chuẩn (PASS - Không lỗi)": (False, None),
            "2. Bo mạch lỗi Hàn Chập (Short Circuit)": (True, "short_circuit"),
            "3. Bo mạch lỗi Mất Lỗ Khoan (Missing Hole)": (True, "missing_hole"),
            "4. Bo mạch lỗi Khuyết Mạch Đồng (Mouse Bite)": (True, "mouse_bite"),
            "5. Bo mạch lỗi Đứt Mạch (Open Circuit)": (True, "open_circuit"),
            "6. Bo mạch lỗi Râu Đồng (Spur)": (True, "spur")
        }

        with col_stream:
            is_inj, def_name = defect_map[sample_type]
            raw_s, gt_s = st.session_state.simulator.generate_pcb_frame(inject_defect=is_inj, specific_defect=def_name)
            ann_s, dets_s, lat_s = st.session_state.detector.infer(raw_s, gt_s)
            st.image(cv2.cvtColor(ann_s, cv2.COLOR_BGR2RGB), caption=f"Mẫu: {sample_type} | Độ trễ ONNX: {lat_s:.1f}ms", use_container_width=True)

            if dets_s:
                st.warning(f"Phát hiện lỗi: **`{dets_s[0]['class'].upper()}`** (Confidence: {dets_s[0]['confidence']*100:.1f}%)")
                if st.button("🧠 Kích Hoạt AI Agent Phân Tích & Tra Cứu SOP Cho Mẫu Này"):
                    incident_res = st.session_state.agent.run(
                        defect_class=dets_s[0]["class"],
                        consecutive_count=3,
                        line_id="SMT-LINE-01"
                    )
                    st.session_state.active_incident = incident_res
                    st.success("Đã gửi thông tin sự cố sang Tab 2 (AI Incident Center)!")
            else:
                st.success("Bo mạch đạt chuẩn chất lượng xuất xưởng.")

# ==========================================
# TAB 2: AI INCIDENT & HITL APPROVAL
# ==========================================
with tab_agent:
    st.subheader("Trung Tâm Xử Lý Sự Cố & Phê Duyệt Hành Động (Human-in-the-Loop)")

    if st.session_state.active_incident:
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
