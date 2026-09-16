import os
import sys
import unittest
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.database import Base, engine, SessionLocal, init_db
from src.backend.models import ProductionLine, InspectionLog, MESTicket
from src.backend.schemas import InspectionCreate
from src.backend.main import record_inspection, resolve_ticket, ActionApprovalRequest

class TestBackendPipeline(unittest.TestCase):
    """Unit tests for SQLite database, telemetry ingestion, and MES action tickets."""

    def setUp(self):
        init_db()
        self.db = SessionLocal()
        # Clean up database tables before each test to guarantee complete test isolation
        self.db.query(InspectionLog).delete()
        self.db.query(MESTicket).delete()
        line = self.db.query(ProductionLine).filter_by(line_id="SMT-LINE-01").first()
        if line:
            line.status = "RUNNING"
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_database_initializes_and_seeds_line(self):
        """Database must have default line SMT-LINE-01 in RUNNING state."""
        line = self.db.query(ProductionLine).filter_by(line_id="SMT-LINE-01").first()
        self.assertIsNotNone(line)
        self.assertEqual(line.status, "RUNNING")

    def test_health_endpoint_returns_ok(self):
        """GET /health must return liveness payload for Docker HEALTHCHECK & smoke tests."""
        from src.backend.main import health_check

        result = health_check()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["service"], "apexinspect-gateway")
        self.assertIn("version", result)

    def test_ensure_thread_id_column_adds_when_missing(self):
        """Migration helper must add missing mes_tickets.thread_id on any engine (Postgres parity)."""
        import os

        from sqlalchemy import create_engine

        from src.backend import database as dbmod

        test_db_path = os.path.join(os.path.dirname(__file__), "_mig_test.db")
        eng = create_engine(
            f"sqlite:///{test_db_path}",
            connect_args={"check_same_thread": False},
        )
        try:
            # Simulate an OLD database created before thread_id existed
            dbmod.Base.metadata.create_all(bind=eng)
            with eng.connect() as conn:
                conn.exec_driver_sql("ALTER TABLE mes_tickets DROP COLUMN thread_id;")
                conn.commit()

            dbmod._ensure_thread_id_column(eng)

            with eng.connect() as conn:
                res = conn.exec_driver_sql("PRAGMA table_info(mes_tickets);")
                cols = [row[1] for row in res.fetchall()]
            self.assertIn("thread_id", cols)
        finally:
            eng.dispose()
            if os.path.exists(test_db_path):
                os.remove(test_db_path)

    def test_record_inspection_triggers_incident_on_three_defects(self):
        """Posting 3 consecutive defects must trigger an incident ticket."""
        line_id = "SMT-LINE-01"
        for i in range(3):
            payload = InspectionCreate(
                line_id=line_id,
                is_defective=True,
                defect_classes=["short_circuit"],
                confidence_scores=[0.92],
                bounding_boxes=[[10, 10, 50, 50]],
                inference_time_ms=25.0
            )
            resp = record_inspection(payload, db=self.db)
            if i == 2:
                self.assertTrue(resp.incident_triggered)
                self.assertIsNotNone(resp.ticket_id)

                ticket = self.db.query(MESTicket).filter_by(ticket_id=resp.ticket_id).first()
                self.assertIsNotNone(ticket)
                self.assertEqual(ticket.action_type, "HALT_LINE")
                self.assertEqual(ticket.status, "PENDING_APPROVAL")

    def test_approve_ticket_halts_production_line(self):
        """Approving a HALT_LINE ticket must set line status to HALTED."""
        test_ticket_id = f"TICK-TEST-{uuid.uuid4().hex[:6]}"
        ticket = MESTicket(
            ticket_id=test_ticket_id,
            line_id="SMT-LINE-01",
            severity="CRITICAL",
            trigger_reason="Test trigger",
            root_cause_analysis="Test RCA",
            recommended_sop="SOP-SMT-001",
            action_type="HALT_LINE",
            status="PENDING_APPROVAL"
        )
        self.db.add(ticket)
        self.db.commit()

        approval_req = ActionApprovalRequest(
            ticket_id=test_ticket_id,
            action="APPROVE",
            approved_by="supervisor_tester"
        )
        resp = resolve_ticket(approval_req, db=self.db)
        self.assertEqual(resp["status"], "EXECUTED")

        line = self.db.query(ProductionLine).filter_by(line_id="SMT-LINE-01").first()
        self.assertEqual(line.status, "HALTED")

    def test_record_inspection_triggers_incident_on_yield_rate_drift(self):
        """Alternating defects exceeding 15% defect rate (Yield Rate Drift) must trigger an incident ticket."""
        line_id = "SMT-LINE-01"
        # 10 items: 6 PASS, 4 DEFECT (40% defect rate > 15% threshold, never 3 in a row)
        pattern = [False, True, False, True, False, True, False, True, False, False]
        last_resp = None
        for is_def in pattern:
            payload = InspectionCreate(
                line_id=line_id,
                is_defective=is_def,
                defect_classes=["mouse_bite"] if is_def else [],
                confidence_scores=[0.85] if is_def else [],
                bounding_boxes=[[10, 10, 30, 30]] if is_def else [],
                inference_time_ms=25.0
            )
            last_resp = record_inspection(payload, db=self.db)

        # After 10 items with 40% defect rate, yield drift should trigger
        self.assertTrue(last_resp.incident_triggered)
        self.assertIsNotNone(last_resp.ticket_id)
        ticket = self.db.query(MESTicket).filter_by(ticket_id=last_resp.ticket_id).first()
        self.assertIsNotNone(ticket)
        self.assertIn("Trượt ngưỡng", ticket.trigger_reason)

    def test_reject_ticket_dismisses_action(self):
        """Rejecting an incident ticket must update status to REJECTED without halting line."""
        test_ticket_id = f"TICK-TEST-REJ-{uuid.uuid4().hex[:6]}"
        ticket = MESTicket(
            ticket_id=test_ticket_id,
            line_id="SMT-LINE-01",
            severity="MEDIUM",
            trigger_reason="Test reject reason",
            root_cause_analysis="Test RCA",
            recommended_sop="SOP-SMT-002",
            action_type="ROUTE_REWORK",
            status="PENDING_APPROVAL"
        )
        self.db.add(ticket)
        self.db.commit()

        approval_req = ActionApprovalRequest(
            ticket_id=test_ticket_id,
            action="REJECT",
            approved_by="supervisor_tester"
        )
        resp = resolve_ticket(approval_req, db=self.db)
        self.assertEqual(resp["status"], "DISMISSED")

        db_ticket = self.db.query(MESTicket).filter_by(ticket_id=test_ticket_id).first()
        self.assertEqual(db_ticket.status, "REJECTED")

if __name__ == "__main__":
    unittest.main()
