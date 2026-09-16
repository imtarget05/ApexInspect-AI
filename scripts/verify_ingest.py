"""Verify ingest results: row counts, defect mix, line metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backend.database import SessionLocal
from src.backend.models import InspectionLog, MESTicket
from src.backend.main import get_line_metrics
from sqlalchemy import text

db = SessionLocal()
try:
    print('total:', db.query(InspectionLog).count())
    print('kaggle:', db.query(InspectionLog).filter(InspectionLog.image_filename.like('pcb-kaggle/%')).count())
    print('defects:', db.query(InspectionLog).filter_by(is_defective=True).count())
    print('tickets:', db.query(MESTicket).count())
    print('kaggle_pass:', db.query(InspectionLog).filter(
        InspectionLog.image_filename.like('pcb-kaggle/%'),
        InspectionLog.is_defective.is_(False)).count())
    # Postgres exposes these JSON columns as `json`, which has no equality
    # operator -> GROUP BY on the raw column fails. Cast to text instead.
    dialect = db.get_bind().dialect.name
    classes_expr = "defect_classes::text" if dialect == "postgresql" else "defect_classes"
    rows = db.execute(text(f"select {classes_expr}, count(*) from inspections "
                           "where is_defective and image_filename like 'pcb-kaggle/%' "
                           "group by 1 order by 2 desc limit 5")).fetchall()
    print('defect_mix:', rows)
    print('lines:', db.execute(text("select line_id, count(*) from inspections "
                                    "where image_filename like 'pcb-kaggle/%' "
                                    "group by 1")).fetchall())
    print('metrics:', get_line_metrics('SMT-LINE-01', db=db))
finally:
    db.close()
