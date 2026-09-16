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

Base = declarative_base()

def _sanitize_db_url(url: str) -> str:
    """Sanitize PostgreSQL URL for maximum driver compatibility."""
    if not url:
        return ""
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    # Remove channel_binding parameter which can fail with psycopg2 on some Linux environments
    if "channel_binding=" in url:
        import re
        url = re.sub(r'[?&]channel_binding=[^&]+', '', url)
        if '?' not in url and '&' in url:
            url = url.replace('&', '?', 1)
    return url.strip()

def _init_engine_and_session():
    """Initializes SQLAlchemy engine with remote Postgres and graceful SQLite fallback."""
    # 1. Read DATABASE_URL from os.environ or streamlit secrets
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
                db_url = str(st.secrets["DATABASE_URL"]).strip()
        except Exception:
            pass

    db_url = _sanitize_db_url(db_url)

    # Explicit SQLite URL (tests use a throwaway file via tests/conftest.py).
    if db_url and db_url.startswith("sqlite"):
        eng = create_engine(db_url, connect_args={"check_same_thread": False})
        print(f"[Database] Using SQLite Engine: {db_url.replace('sqlite:///', '')}")
        return eng, sessionmaker(autocommit=False, autoflush=False, bind=eng)

    if db_url and db_url.startswith("postgresql"):
        try:
            import psycopg2  # noqa: F401
            eng = create_engine(
                db_url,
                pool_pre_ping=True,
                connect_args={"connect_timeout": 5}
            )
            # Lightweight verification ping
            with eng.connect() as conn:
                pass
            print("[Database] Successfully connected to Remote PostgreSQL (Neon)")
            return eng, sessionmaker(autocommit=False, autoflush=False, bind=eng)
        except Exception as err:
            print(f"[Database] Remote PostgreSQL connection failed: {err}. Falling back to SQLite.")

    # Fallback to local SQLite engine
    sqlite_url = "sqlite:///./factory.db"
    eng = create_engine(sqlite_url, connect_args={"check_same_thread": False})
    print("[Database] Using Local SQLite Engine: factory.db")
    return eng, sessionmaker(autocommit=False, autoflush=False, bind=eng)

engine, SessionLocal = _init_engine_and_session()

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
    """Initializes tables, ensures schema migrations, and seeds default line configuration."""
    from .models import ProductionLine
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as err:
        print(f"[Database] Metadata create_all note: {err}")

    # Seamless SQLite schema migrations for existing local databases
    try:
        with engine.connect() as conn:
            res = conn.exec_driver_sql("PRAGMA table_info(mes_tickets);")
            cols = [row[1] for row in res.fetchall()]
            if cols and "thread_id" not in cols:
                conn.exec_driver_sql("ALTER TABLE mes_tickets ADD COLUMN thread_id VARCHAR(100);")
                conn.commit()
    except Exception as mig_err:
        pass

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
