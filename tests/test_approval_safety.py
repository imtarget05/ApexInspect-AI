"""Acceptance regression tests for incident triggering and HITL approval safety.

Covers the four acceptance criteria agreed for the stability closure:

1. Yield-drift/consecutive-defect triggering still creates an incident ticket,
   and a **second** cascade does not create a duplicate while one ticket is
   still ``PENDING_APPROVAL``.
2. An approved action dispatches to the PLC (simulated double) exactly once.
3. A **rejected** action must NOT dispatch anything to the PLC and must leave the
   production line untouched (the "Reject => no execution" guarantee).
4. All tests use a recording PLC double / loopback simulator only - no test may
   emit a command to real hardware.
"""
import os
import sys
import unittest
import uuid
from collections import namedtuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Return shape of InspectionService.record_telemetry: (log, incident_triggered, ticket_id)
CascadeResult = namedtuple("CascadeResult", "incident_triggered ticket_id")

from src.backend.database import SessionLocal, init_db
from src.backend.models import AuditLog, InspectionLog, MESTicket, ProductionLine
from src.backend.schemas import InspectionCreate
from src.backend.service import InspectionService

LINE_ID = "SMT-LINE-01"


class RecordingPLCBridge:
    """Simulated PLC double: records every actuation instead of touching hardware."""

    def __init__(self):
        self.calls = []

    def halt_line(self, line_id):
        self.calls.append(("halt_line", line_id))
        return {"action": "HALT_LINE", "status": "SIMULATED", "conveyor_running": False}

    def resume_line(self, line_id):
        self.calls.append(("resume_line", line_id))
        return {"action": "RESUME_LINE", "status": "SIMULATED", "conveyor_running": True}

    def divert_rework(self, line_id):
        self.calls.append(("divert_rework", line_id))
        return {"action": "ROUTE_REWORK", "status": "SIMULATED", "tower_light": "YELLOW"}


class ApprovalSafetyTestBase(unittest.TestCase):
    def setUp(self):
        init_db()
        self.db = SessionLocal()
        self.db.query(InspectionLog).delete()
        self.db.query(MESTicket).delete()
        line = self.db.query(ProductionLine).filter_by(line_id=LINE_ID).first()
        if line:
            line.status = "RUNNING"
        self.db.commit()

    def tearDown(self):
        self.db.close()

    # --- helpers ---------------------------------------------------------
    def _record(self, defect_class="short_circuit"):
        """Ingest one defective item through the service layer.

        ``agent=None`` keeps the test hermetic: no LangGraph run, no LLM network
        call, no rate-limit flakiness.
        """
        payload = InspectionCreate(
            line_id=LINE_ID,
            is_defective=True,
            defect_classes=[defect_class],
            confidence_scores=[0.93],
            bounding_boxes=[[10, 10, 50, 50]],
            inference_time_ms=24.0,
        )
        _, incident_triggered, ticket_id = InspectionService.record_telemetry(
            db=self.db, payload=payload, agent=None
        )
        return incident_triggered, ticket_id

    def _three_consecutive_defects(self):
        """Record 3 consecutive defective items; returns this cascade's outcome.

        Trigger Condition 1 fires as soon as the sliding window holds 3 defective
        items, which can be the 1st record of a cascade (the window still carries
        defects from an earlier cascade) or the 3rd. The returned value therefore
        aggregates the whole cascade:

        * ``incident_triggered`` - True if any record opened an incident ticket
        * ``ticket_id``         - the ticket opened by this cascade, or None when
          the pending-ticket duplicate guard suppressed it
        """
        incident_triggered = False
        ticket_id = None
        for _ in range(3):
            triggered, created_id = self._record()
            incident_triggered = incident_triggered or triggered
            if created_id and ticket_id is None:
                ticket_id = created_id
        return CascadeResult(incident_triggered, ticket_id)

    def _resolve(self, ticket_id, action, plc_bridge=None):
        """Runs the service-layer resolution (never the FastAPI endpoint)."""
        return InspectionService.resolve_ticket(
            db=self.db,
            ticket_id=ticket_id,
            action=action,
            approved_by="supervisor_tester",
            agent=None,
            plc_bridge=plc_bridge,
        )

    def _make_pending_ticket(self, action_type="HALT_LINE"):
        ticket_id = f"TICK-TEST-SAFE-{uuid.uuid4().hex[:6]}"
        self.db.add(
            MESTicket(
                ticket_id=ticket_id,
                line_id=LINE_ID,
                severity="CRITICAL",
                trigger_reason="Safety regression test ticket",
                root_cause_analysis="Test RCA",
                recommended_sop="SOP-SMT-001",
                action_type=action_type,
                status="PENDING_APPROVAL",
            )
        )
        self.db.commit()
        return ticket_id

    def _line(self):
        self.db.expire_all()
        return self.db.query(ProductionLine).filter_by(line_id=LINE_ID).first()

    # --- 1. duplicate-ticket prevention ---------------------------------
    def test_duplicate_pending_ticket_is_not_created(self):
        """A second defect cascade must not open a second PENDING_APPROVAL ticket."""
        first = self._three_consecutive_defects()
        self.assertTrue(first.incident_triggered)
        self.assertIsNotNone(first.ticket_id)

        second = self._three_consecutive_defects()
        self.assertFalse(second.incident_triggered)
        self.assertIsNone(second.ticket_id)

        pending = (
            self.db.query(MESTicket)
            .filter_by(line_id=LINE_ID, status="PENDING_APPROVAL")
            .all()
        )
        self.assertEqual(len(pending), 1, "duplicate pending ticket was created")

    def test_new_ticket_allowed_after_previous_one_resolved(self):
        """Once the pending ticket is resolved, the next cascade may open a new one."""
        first = self._three_consecutive_defects()
        self.assertTrue(first.incident_triggered)

        self._resolve(first.ticket_id, "REJECT")

        second = self._three_consecutive_defects()
        self.assertTrue(second.incident_triggered)
        self.assertIsNotNone(second.ticket_id)
        self.assertNotEqual(first.ticket_id, second.ticket_id)

    # --- 2. approve dispatches to simulated PLC --------------------------
    def test_approve_dispatches_halt_once_to_simulated_plc(self):
        ticket_id = self._make_pending_ticket(action_type="HALT_LINE")
        plc = RecordingPLCBridge()

        resp = self._resolve(ticket_id, "APPROVE", plc)

        self.assertEqual(resp["status"], "EXECUTED")
        self.assertTrue(resp["plc_dispatched"])
        self.assertEqual(plc.calls, [("halt_line", LINE_ID)], "PLC dispatch count mismatch")
        self.assertEqual(self._line().status, "HALTED")

    # --- 3. reject must not execute anything -----------------------------
    def test_reject_does_not_dispatch_to_plc_and_keeps_line_running(self):
        ticket_id = self._make_pending_ticket(action_type="HALT_LINE")
        plc = RecordingPLCBridge()

        resp = self._resolve(ticket_id, "REJECT", plc)

        self.assertEqual(resp["status"], "DISMISSED")
        self.assertFalse(resp["plc_dispatched"])
        self.assertEqual(plc.calls, [], "rejected action reached the PLC")
        self.assertEqual(self._line().status, "RUNNING", "rejected action halted the line")

        ticket = self.db.query(MESTicket).filter_by(ticket_id=ticket_id).first()
        self.assertEqual(ticket.status, "REJECTED")
        self.assertIsNotNone(ticket.resolved_at)

    def test_reject_routes_rework_ticket_without_plc_dispatch(self):
        """ROUTE_REWORK tickets must also stay untouched when rejected."""
        ticket_id = self._make_pending_ticket(action_type="ROUTE_REWORK")
        plc = RecordingPLCBridge()

        resp = self._resolve(ticket_id, "REJECT", plc)

        self.assertEqual(resp["status"], "DISMISSED")
        self.assertEqual(plc.calls, [])
        self.assertEqual(self._line().status, "RUNNING")

    def test_unknown_ticket_is_rejected_without_actuation(self):
        plc = RecordingPLCBridge()
        resp = self._resolve("TICK-DOES-NOT-EXIST", "APPROVE", plc)
        self.assertEqual(resp["status"], "ERROR")
        self.assertEqual(plc.calls, [])

    # --- 4. no test may touch real hardware ------------------------------
    def test_default_plc_path_is_injected_and_simulated(self):
        """`resolve_ticket` only actuates the injected bridge; default stays simulated."""
        from src.industrial.plc_bridge import PLCBridge

        # Unreachable loopback port => simulation fallback, never a real device.
        default_bridge = PLCBridge(
            host="127.0.0.1", port=59998, timeout=0.2, simulation_fallback=True
        )
        self.assertFalse(default_bridge.is_connected)
        self.assertEqual(default_bridge.halt_line(LINE_ID)["status"], "SIMULATED")

        # With no bridge injected the service must not actuate anything at all.
        ticket_id = self._make_pending_ticket(action_type="HALT_LINE")
        resp = self._resolve(ticket_id, "APPROVE", plc_bridge=None)
        self.assertEqual(resp["status"], "EXECUTED")
        self.assertFalse(resp["plc_dispatched"])

    # --- 5. idempotent replay (Plan 03) ----------------------------------
    def test_replay_same_key_returns_stored_response_without_second_dispatch(self):
        """Same ticket + same key: replay returns the stored EXECUTED
        response; the PLC bridge is touched exactly once."""
        ticket_id = self._make_pending_ticket(action_type="HALT_LINE")
        plc = RecordingPLCBridge()
        key = f"replay-{uuid.uuid4().hex[:6]}"

        def resolve():
            return InspectionService.resolve_ticket(
                db=self.db,
                ticket_id=ticket_id,
                action="APPROVE",
                approved_by="supervisor_tester",
                agent=None,
                plc_bridge=plc,
                idempotency_key=key,
            )

        first = resolve()
        second = resolve()

        self.assertEqual(first["status"], "EXECUTED")
        self.assertEqual(second["status"], "EXECUTED")
        self.assertEqual(second["ticket_id"], ticket_id)
        self.assertEqual(plc.calls, [("halt_line", LINE_ID)])


if __name__ == "__main__":
    unittest.main()
