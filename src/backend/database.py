import os
import uuid
import json
import random
import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables from .env
load_dotenv()

# 1. Read Neon PostgreSQL URL or fallback to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./factory.db"
    connect_args = {"check_same_thread": False}
    print("[Database] Using Local SQLite Engine: factory.db")
else:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    connect_args = {}
    print("[Database] Connected to Remote Managed Database")

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency for obtaining database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def seed_demo_data(force: bool = False):
    """Populates the database with realistic sample inspection logs and MES tickets."""
    from .models import ProductionLine, InspectionLog, MESTicket
    db = SessionLocal()
    try:
        total_logs = db.query(InspectionLog).count()
        if total_logs > 0 and not force:
            return

        now = datetime.datetime.now(datetime.timezone.utc)
        defect_samples = [
            ("short_circuit", [0.94], [[160, 220, 240, 280]]),
            ("missing_hole", [0.88], [[110, 118, 130, 138]]),
            ("mouse_bite", [0.85], [[238, 125, 258, 135]]),
            ("open_circuit", [0.92], [[305, 175, 335, 185]]),
            ("spur", [0.81], [[275, 228, 298, 245]])
        ]

        # 1. Seed 35 PASS items
        for i in range(35):
            insp_id = f"INSP-PASS-{uuid.uuid4().hex[:6].upper()}"
            t = now - datetime.timedelta(minutes=random.randint(5, 180))
            db.add(InspectionLog(
                inspection_id=insp_id,
                line_id="SMT-LINE-01",
                is_defective=False,
                defect_classes=[],
                confidence_scores=[],
                bounding_boxes=[],
                inference_time_ms=round(random.uniform(24.0, 32.0), 1),
                timestamp=t
            ))

        # 2. Seed 15 DEFECT items with diverse classes
        for i in range(15):
            cls_name, confs, bboxes = random.choice(defect_samples)
            insp_id = f"INSP-DEF-{uuid.uuid4().hex[:6].upper()}"
            t = now - datetime.timedelta(minutes=random.randint(1, 150))
            db.add(InspectionLog(
                inspection_id=insp_id,
                line_id="SMT-LINE-01",
                is_defective=True,
                defect_classes=[cls_name],
                confidence_scores=confs,
                bounding_boxes=bboxes,
                inference_time_ms=round(random.uniform(25.0, 35.0), 1),
                timestamp=t
            ))

        # 3. Seed 2 MES Incident Tickets
        t1_id = f"TICK-{now.strftime('%Y%m%d')}-001"
        if not db.query(MESTicket).filter_by(ticket_id=t1_id).first():
            db.add(MESTicket(
                ticket_id=t1_id,
                line_id="SMT-LINE-01",
                severity="CRITICAL",
                trigger_reason="Phát hiện 3 sản phẩm liên tiếp lỗi short_circuit tại trạm dán kem hàn.",
                root_cause_analysis="Tấm Stencil bị đọng cặn thiếc hàn dư thừa sau 4 giờ vận hành liên tục (theo SOP-SMT-001). Cần làm sạch bằng cồn IPA.",
                recommended_sop="SOP-SMT-001-solder-bridge.md",
                action_type="HALT_LINE",
                status="APPROVED",
                approved_by="supervisor_tan",
                created_at=now - datetime.timedelta(hours=2),
                resolved_at=now - datetime.timedelta(hours=1, minutes=45)
            ))

        t2_id = f"TICK-{now.strftime('%Y%m%d')}-002"
        if not db.query(MESTicket).filter_by(ticket_id=t2_id).first():
            db.add(MESTicket(
                ticket_id=t2_id,
                line_id="SMT-LINE-01",
                severity="MEDIUM",
                trigger_reason="Phát hiện 2 sản phẩm liên tiếp lỗi missing_hole.",
                root_cause_analysis="Đầu hút chân không bị bám bụi bẩn (theo SOP-SMT-002). Đề xuất chuyển trạm rework.",
                recommended_sop="SOP-SMT-002-missing-component.md",
                action_type="ROUTE_REWORK",
                status="PENDING_APPROVAL",
                created_at=now - datetime.timedelta(minutes=25)
            ))

        db.commit()
        print("[Database] Seeded demo inspection logs & MES tickets successfully.")
    except Exception as e:
        print(f"[Database] Seed note: {e}")
        db.rollback()
    finally:
        db.close()

def init_db():
    """Initializes tables and seeds default line configuration."""
    from .models import ProductionLine
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        existing_line = db.query(ProductionLine).filter_by(line_id="SMT-LINE-01").first()
        if not existing_line:
            default_line = ProductionLine(
                line_id="SMT-LINE-01",
                name="Surface Mount Assembly Line 01",
                status="RUNNING",
                current_product="Automotive ECU Rev 3.2",
                target_yield_rate=95.0
            )
            db.add(default_line)
            db.commit()
            print("[Database] Seeded initial production line: SMT-LINE-01")
    except Exception as e:
        print(f"[Database] Initialization note: {e}")
        db.rollback()
    finally:
        db.close()
