import uuid
import datetime
import json
from collections import Counter
from typing import Dict, Any, Tuple, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import desc, update
from sqlalchemy.exc import IntegrityError

from .models import ProductionLine, InspectionLog, MESTicket, AuditLog
from .models import IdempotencyKey, OutboxEvent
from .schemas import InspectionCreate, ActionApprovalRequest, LineMetricsResponse


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


class InspectionService:
    """
    Unified Industrial Quality & MES Service Layer.
    Acts as the Single Source of Truth for:
    - Edge inspection telemetry ingestion
    - Rule-based Incident Triggers (3-consecutive defects & Yield Drift)
    - LangGraph HITL Agent invocation with isolated thread IDs
    - Supervisor ticket resolution & PLC hardware dispatch
    - Immutable audit logging
    """

    @staticmethod
    def record_telemetry(
        db: Session,
        payload: InspectionCreate,
        agent: Optional[Any] = None
    ) -> Tuple[InspectionLog, bool, Optional[str]]:
        """Ingests telemetry, evaluates triggers, and dispatches to LangGraph if threshold met."""
        insp_id = f"INSP-{uuid.uuid4().hex[:8].upper()}"
        log_entry = InspectionLog(
            inspection_id=insp_id,
            line_id=payload.line_id,
            is_defective=payload.is_defective,
            defect_classes=payload.defect_classes,
            confidence_scores=payload.confidence_scores,
            bounding_boxes=payload.bounding_boxes,
            inference_time_ms=payload.inference_time_ms,
            timestamp=utc_now()
        )
        db.add(log_entry)
        db.commit()

        # Query recent logs in sliding window of up to 30 items
        recent_logs = db.query(InspectionLog)\
            .filter(InspectionLog.line_id == payload.line_id)\
            .order_by(desc(InspectionLog.timestamp))\
            .limit(30)\
            .all()

        incident_triggered = False
        created_ticket_id = None

        # Concurrency safety: check for existing pending ticket with row-level lock where supported
        query = db.query(MESTicket).filter(
            MESTicket.line_id == payload.line_id,
            MESTicket.status == "PENDING_APPROVAL"
        )
        try:
            if db.bind and db.bind.dialect.name == "postgresql":
                existing_pending = query.with_for_update().first()
            else:
                existing_pending = query.first()
        except Exception:
            existing_pending = query.first()

        if not existing_pending and recent_logs:
            now_utc = utc_now()
            ticket_id = f"TICK-{now_utc.strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
            thread_id = f"incident-{payload.line_id}-{ticket_id}"

            # Trigger Condition 1: 3 consecutive defective items
            recent_3 = recent_logs[:3]
            consecutive_triggered = len(recent_3) == 3 and all(item.is_defective for item in recent_3)

            # Trigger Condition 2: Yield Rate Drift (error rate > 15% in window of >= 10 items)
            window_len = len(recent_logs)
            window_defects = sum(1 for item in recent_logs if item.is_defective)
            defect_ratio = window_defects / window_len if window_len > 0 else 0.0
            drift_triggered = window_len >= 10 and defect_ratio > 0.15

            if consecutive_triggered:
                defects_found = payload.defect_classes[0] if payload.defect_classes else "short_circuit"
                agent_result = {}
                if agent is not None:
                    agent_result = agent.run(
                        defect_class=defects_found,
                        consecutive_count=3,
                        line_id=payload.line_id,
                        thread_id=thread_id
                    )

                ticket = MESTicket(
                    ticket_id=ticket_id,
                    line_id=payload.line_id,
                    severity="CRITICAL" if defects_found in ["short_circuit", "short"] else "MEDIUM",
                    trigger_reason=f"Phát hiện 3 sản phẩm liên tiếp có lỗi '{defects_found}' trên chuyền {payload.line_id}.",
                    root_cause_analysis=agent_result.get("rca_analysis", "Tự động kích hoạt do phát hiện chuỗi lỗi liên tiếp."),
                    recommended_sop=", ".join(agent_result.get("sop_citations", [])) or "SOP-SMT-001",
                    action_type=agent_result.get("proposed_action", "HALT_LINE"),
                    status="PENDING_APPROVAL",
                    thread_id=thread_id,
                    created_at=now_utc
                )
                db.add(ticket)

                audit = AuditLog(
                    log_id=f"AUD-{uuid.uuid4().hex[:8].upper()}",
                    operator_id="AI_INCIDENT_AGENT",
                    action="TRIGGER_CONSECUTIVE_DEFECTS",
                    line_id=payload.line_id,
                    ticket_id=ticket_id,
                    details=f"Created incident ticket {ticket_id} for defect '{defects_found}'."
                )
                db.add(audit)
                db.commit()

                incident_triggered = True
                created_ticket_id = ticket_id

            elif drift_triggered:
                all_def_types = []
                for item in recent_logs:
                    if item.is_defective and item.defect_classes:
                        cls_list = item.defect_classes if isinstance(item.defect_classes, list) else [item.defect_classes]
                        all_def_types.extend(cls_list)
                primary_defect = Counter(all_def_types).most_common(1)[0][0] if all_def_types else "defect"

                agent_result = {}
                if agent is not None:
                    agent_result = agent.run(
                        defect_class=primary_defect,
                        consecutive_count=window_defects,
                        line_id=payload.line_id,
                        thread_id=thread_id
                    )

                ticket = MESTicket(
                    ticket_id=ticket_id,
                    line_id=payload.line_id,
                    severity="CRITICAL" if defect_ratio > 0.30 else "MEDIUM",
                    trigger_reason=f"Trượt ngưỡng tỷ lệ lỗi (Yield Rate Drift): {defect_ratio*100:.1f}% lỗi ({window_defects}/{window_len}) trong cửa sổ trượt (ngưỡng cho phép: 15%). Lỗi xuất hiện nhiều nhất: '{primary_defect}'.",
                    root_cause_analysis=agent_result.get("rca_analysis", "Tự động kích hoạt do tỷ lệ lỗi vượt ngưỡng 15%."),
                    recommended_sop=", ".join(agent_result.get("sop_citations", [])) or "SOP-SMT-001",
                    action_type=agent_result.get("proposed_action", "ROUTE_REWORK"),
                    status="PENDING_APPROVAL",
                    thread_id=thread_id,
                    created_at=now_utc
                )
                db.add(ticket)

                audit = AuditLog(
                    log_id=f"AUD-{uuid.uuid4().hex[:8].upper()}",
                    operator_id="AI_INCIDENT_AGENT",
                    action="TRIGGER_YIELD_DRIFT",
                    line_id=payload.line_id,
                    ticket_id=ticket_id,
                    details=f"Created incident ticket {ticket_id} for yield drift ({defect_ratio*100:.1f}%)."
                )
                db.add(audit)
                db.commit()

                incident_triggered = True
                created_ticket_id = ticket_id

        return log_entry, incident_triggered, created_ticket_id

    @staticmethod
    def resolve_ticket(
        db: Session,
        ticket_id: str,
        action: str,
        approved_by: str,
        agent: Optional[Any] = None,
        plc_bridge: Optional[Any] = None,
        idempotency_key: Optional[str] = None,
        source_ip: str = "127.0.0.1"
    ) -> Dict[str, Any]:
        """Supervises ticket approval/rejection, resumes LangGraph HITL, and dispatches to PLC."""
        ticket = db.query(MESTicket).filter(MESTicket.ticket_id == ticket_id).first()
        if not ticket:
            return {"status": "ERROR", "message": f"Ticket {ticket_id} not found."}

        action = action.upper()
        if action not in {"APPROVE", "REJECT"}:
            return {"status": "ERROR", "message": "action must be APPROVE or REJECT"}

        now_utc = utc_now()
        is_approved = action == "APPROVE"
        resolution_key = idempotency_key or f"server-{uuid.uuid4().hex}"
        idem_scope = f"mes_resolve:{ticket_id}"

        # Idempotency gate (Plan 03): a replayed key returns the stored
        # response without touching the ticket, agent, or PLC. Scope is
        # per-ticket so a key can never resolve a different ticket.
        if idempotency_key:
            prior = (
                db.query(IdempotencyKey)
                .filter_by(caller_scope=idem_scope, idem_key=idempotency_key)
                .first()
            )
            if prior is not None and prior.status == "completed":
                return json.loads(prior.response)

        # Reserve the ticket before hardware is touched. The conditional update
        # lets only one request leave PENDING_APPROVAL, even across sessions.
        reserved = db.execute(
            update(MESTicket)
            .where(
                MESTicket.ticket_id == ticket_id,
                MESTicket.status == "PENDING_APPROVAL",
            )
            .values(
                status="RESOLVING",
                approved_by=approved_by,
                resolution_key=resolution_key,
            )
        )
        if reserved.rowcount != 1:
            db.rollback()
            return {"status": "CONFLICT", "message": "Ticket has already been resolved or is being resolved."}
        db.commit()
        db.refresh(ticket)

        if not is_approved:
            plc_result = {"status": "NOT_DISPATCHED", "reason": "REJECTED_BY_SUPERVISOR"}
            ticket.status = "REJECTED"
            ticket.plc_status = "NOT_DISPATCHED"
            ticket.plc_result_json = json.dumps(plc_result, sort_keys=True)
            ticket.resolved_at = now_utc
            plc_dispatched = False
        else:
            if plc_bridge is None and ticket.thread_id:
                # Agent-driven HITL flow (persisted graph thread): the service
                # is allowed to use the explicit SIMULATION bridge — never a
                # hardware connection. Direct calls without a thread stay
                # fail-closed (ACTUATION_FAILED) so nothing is silently
                # actuated outside the supervised flow.
                try:
                    from ..industrial.plc_bridge import PLCBridge

                    plc_bridge = PLCBridge(mode="simulation")
                except Exception:
                    plc_bridge = None
            if plc_bridge is None:
                plc_result = {"status": "FAILED", "error": "PLC_BRIDGE_UNAVAILABLE"}
            else:
                try:
                    if ticket.action_type == "HALT_LINE":
                        plc_result = plc_bridge.halt_line(ticket.line_id)
                    elif ticket.action_type == "ROUTE_REWORK":
                        plc_result = plc_bridge.divert_rework(ticket.line_id)
                    else:
                        plc_result = {"status": "FAILED", "error": "UNSUPPORTED_ACTION_TYPE"}
                except Exception as exc:
                    plc_result = {"status": "FAILED", "error": str(exc)}

            plc_status = plc_result.get("status", "FAILED")
            plc_dispatched = plc_status in {"DISPATCHED", "SIMULATED"}
            ticket.plc_status = plc_status
            ticket.plc_result_json = json.dumps(plc_result, sort_keys=True)
            ticket.resolved_at = now_utc
            if plc_dispatched:
                ticket.status = "EXECUTED"
                line = db.query(ProductionLine).filter(ProductionLine.line_id == ticket.line_id).first()
                if line and ticket.action_type == "HALT_LINE":
                    line.status = "HALTED"
            else:
                ticket.status = "ACTUATION_FAILED"

        # 3. Resume LangGraph HITL checkpoint if thread_id exists.
        # The ticket reservation already succeeded, so resume that exact
        # persisted graph with the durable PLC outcome as evidence.
        agent_resumed = False
        if agent is not None and ticket.thread_id:
            try:
                import json as _json

                try:
                    _plc_evidence = _json.loads(ticket.plc_result_json or "{}")
                except Exception:
                    _plc_evidence = {"status": ticket.plc_status or "UNKNOWN"}
                agent.resume_approval(
                    thread_id=ticket.thread_id,
                    approved=is_approved,
                    supervisor_id=approved_by,
                    plc_result=_plc_evidence,
                    ticket_id=ticket.ticket_id,
                )
                agent_resumed = True
            except Exception as e:
                print(f"[Service] Note during LangGraph resume: {e}")

        # 4. Record Audit Log
        audit = AuditLog(
            log_id=f"AUD-{uuid.uuid4().hex[:8].upper()}",
            operator_id=approved_by,
            action="APPROVE_ACTION" if is_approved else "REJECT_ACTION",
            line_id=ticket.line_id,
            ticket_id=ticket.ticket_id,
            details=(
                f"Action {ticket.action_type} {'approved' if is_approved else 'rejected'} by {approved_by}. "
                f"(PLC={ticket.plc_status}, AgentResumed={agent_resumed})"
            ),
            source_ip=source_ip
        )
        db.add(audit)
        db.commit()

        if is_approved and ticket.status == "EXECUTED":
            result = {
                "status": "EXECUTED",
                "message": f"Dây chuyền {line.line_id if line else ticket.line_id} đã được thi hành lệnh {ticket.action_type}.",
                "ticket_id": ticket.ticket_id,
                "plc_dispatched": plc_dispatched,
                "agent_resumed": agent_resumed
            }
        elif is_approved:
            result = {
                "status": "ACTUATION_FAILED",
                "message": f"PLC command for ticket {ticket.ticket_id} did not execute.",
                "ticket_id": ticket.ticket_id,
                "plc_dispatched": False,
                "plc_result": plc_result,
                "agent_resumed": agent_resumed,
            }
        else:
            result = {
                "status": "DISMISSED",
                "message": f"Lệnh can thiệp cho ticket {ticket.ticket_id} đã bị từ chối bởi Quản đốc {approved_by}.",
                "ticket_id": ticket.ticket_id,
                "plc_dispatched": False,
                "agent_resumed": agent_resumed
            }

        # Record the idempotent outcome + outbox event in one transaction so
        # a replay returns exactly this response. Lost races (IntegrityError)
        # mean another worker completed first: return the stored response.
        if idempotency_key:
            try:
                with db.begin_nested():
                    db.add(IdempotencyKey(
                        caller_scope=idem_scope,
                        idem_key=idempotency_key,
                        fingerprint=ticket_id,
                        status="completed",
                        response=json.dumps(result, default=str),
                        expires_at="",
                    ))
                    db.add(OutboxEvent(
                        event_id=f"mes-resolve:{ticket_id}:{resolution_key}",
                        destination="mes-resolve",
                        payload=json.dumps(
                            {"ticket_id": ticket_id,
                             "outcome": result["status"]},
                            default=str,
                        ),
                        version="v1",
                    ))
            except IntegrityError:
                db.rollback()
                prior = (
                    db.query(IdempotencyKey)
                    .filter_by(caller_scope=idem_scope, idem_key=idempotency_key)
                    .first()
                )
                return json.loads(prior.response)
            db.commit()
        return result

    @staticmethod
    def resume_line(
        db: Session,
        line_id: str,
        operator_id: str = "supervisor",
        plc_bridge: Optional[Any] = None,
        source_ip: str = "127.0.0.1"
    ) -> bool:
        """Safely restarts a halted line and synchronizes state to DB & PLC."""
        line = db.query(ProductionLine).filter(ProductionLine.line_id == line_id).first()
        if line:
            line.status = "RUNNING"
            line.updated_at = utc_now()

            if plc_bridge is not None:
                try:
                    plc_bridge.resume_line(line_id)
                except Exception as e:
                    print(f"[Service] Note during PLC resume: {e}")

            audit = AuditLog(
                log_id=f"AUD-{uuid.uuid4().hex[:8].upper()}",
                operator_id=operator_id,
                action="RESUME_LINE",
                line_id=line_id,
                details=f"Production line {line_id} resumed to RUNNING state.",
                source_ip=source_ip
            )
            db.add(audit)
            db.commit()
            return True
        return False
