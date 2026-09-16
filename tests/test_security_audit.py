import os
import sys
import unittest
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.database import SessionLocal, init_db
from src.backend.models import ProductionLine, InspectionLog, MESTicket, AuditLog
from src.backend.schemas import InspectionCreate
from src.backend.service import InspectionService
from src.agent.graph import QualityIncidentAgent

class TestSecurityAndAudit(unittest.TestCase):
    """Verifies audit logging, dynamic thread_id generation, and secure resolution."""

    def setUp(self):
        init_db()
        self.db = SessionLocal()
        self.db.query(AuditLog).delete()
        self.db.query(MESTicket).delete()
        self.db.query(InspectionLog).delete()
        line = self.db.query(ProductionLine).filter_by(line_id="SMT-LINE-01").first()
        if line:
            line.status = "RUNNING"
        self.db.commit()
        self.agent = QualityIncidentAgent()

    def tearDown(self):
        self.db.close()

    def test_audit_log_created_on_incident_trigger(self):
        """InspectionService must write an immutable AuditLog entry when triggers fire."""
        line_id = "SMT-LINE-01"
        last_ticket_id = None
        for i in range(3):
            payload = InspectionCreate(
                line_id=line_id,
                is_defective=True,
                defect_classes=["short_circuit"],
                confidence_scores=[0.95],
                bounding_boxes=[[10, 10, 50, 50]],
                inference_time_ms=28.0
            )
            _, triggered, ticket_id = InspectionService.record_telemetry(
                db=self.db,
                payload=payload,
                agent=self.agent
            )
            if triggered:
                last_ticket_id = ticket_id

        self.assertIsNotNone(last_ticket_id)
        # Verify MESTicket has a dynamic thread_id
        ticket = self.db.query(MESTicket).filter_by(ticket_id=last_ticket_id).first()
        self.assertIsNotNone(ticket)
        self.assertTrue(ticket.thread_id.startswith(f"incident-{line_id}-"))

        # Verify AuditLog recorded
        audit = self.db.query(AuditLog).filter_by(ticket_id=last_ticket_id).first()
        self.assertIsNotNone(audit)
        self.assertEqual(audit.action, "TRIGGER_CONSECUTIVE_DEFECTS")
        self.assertEqual(audit.operator_id, "AI_INCIDENT_AGENT")

    def test_audit_log_and_agent_resume_on_supervisor_approval(self):
        """Resolving a ticket must record an audit entry and resume LangGraph state."""
        line_id = "SMT-LINE-01"
        for _ in range(3):
            payload = InspectionCreate(
                line_id=line_id,
                is_defective=True,
                defect_classes=["short_circuit"],
                confidence_scores=[0.95],
                bounding_boxes=[[10, 10, 50, 50]],
                inference_time_ms=28.0
            )
            InspectionService.record_telemetry(db=self.db, payload=payload, agent=self.agent)

        ticket = self.db.query(MESTicket).filter_by(status="PENDING_APPROVAL").first()
        self.assertIsNotNone(ticket)

        resolve_res = InspectionService.resolve_ticket(
            db=self.db,
            ticket_id=ticket.ticket_id,
            action="APPROVE",
            approved_by="supervisor_kpi",
            agent=self.agent,
            source_ip="192.168.1.100"
        )
        self.assertEqual(resolve_res["status"], "EXECUTED")
        self.assertTrue(resolve_res.get("agent_resumed"))

        # Verify AuditLog for supervisor sign-off
        audit_approve = self.db.query(AuditLog).filter_by(
            ticket_id=ticket.ticket_id,
            action="APPROVE_ACTION"
        ).first()
        self.assertIsNotNone(audit_approve)
        self.assertEqual(audit_approve.operator_id, "supervisor_kpi")
        self.assertEqual(audit_approve.source_ip, "192.168.1.100")

    def test_api_key_authentication_enforced(self):
        """MES action endpoint must reject requests without valid X-API-KEY."""
        from fastapi.testclient import TestClient
        from src.backend.main import app

        client = TestClient(app)
        # 1. Missing header -> 401
        res_no_key = client.post("/api/v1/mes/action", json={"ticket_id": "TICK-TEST", "action": "APPROVE"})
        self.assertEqual(res_no_key.status_code, 401)

        # 2. Invalid header -> 403
        res_bad_key = client.post(
            "/api/v1/mes/action",
            json={"ticket_id": "TICK-TEST", "action": "APPROVE"},
            headers={"X-API-KEY": "wrong-secret-key"}
        )
        self.assertEqual(res_bad_key.status_code, 403)

        # 3. Valid header with non-existent ticket -> 404 (passed auth successfully)
        res_valid_key = client.post(
            "/api/v1/mes/action",
            json={"ticket_id": "NON-EXISTENT-TICKET", "action": "APPROVE"},
            headers={"X-API-KEY": "dev-factory-key-secret"}
        )
        self.assertEqual(res_valid_key.status_code, 404)

if __name__ == "__main__":
    unittest.main()
