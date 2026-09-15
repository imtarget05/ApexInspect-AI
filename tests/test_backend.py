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

if __name__ == "__main__":
    unittest.main()
