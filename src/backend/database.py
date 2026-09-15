import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. Read Neon PostgreSQL URL or fallback to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./factory.db"
    connect_args = {"check_same_thread": False}
    print("[Database] Using Local SQLite Engine: factory.db")
else:
    # Ensure standard postgresql:// schema
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
