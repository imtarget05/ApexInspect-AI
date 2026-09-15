import uuid
import datetime
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc

from .database import get_db, init_db
from .models import ProductionLine, InspectionLog, MESTicket
from .schemas import InspectionCreate, InspectionResponse, ActionApprovalRequest, LineMetricsResponse
from ..agent.graph import QualityIncidentAgent

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="ApexInspect AI Gateway",
    description="Industrial Telemetry Ingestion, Automated Defect Resolution & MES Synchronization",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared agent instance for incident resolution
incident_agent = QualityIncidentAgent()

def get_current_utc():
    return datetime.datetime.now(datetime.timezone.utc)

@app.post("/api/v1/inspections", response_model=InspectionResponse)
def record_inspection(payload: InspectionCreate, db: Session = Depends(get_db)):
    """
    Ingests defect telemetry from edge camera stream, checks incident triggers,
    and invokes LangGraph Quality Incident Agent upon detecting defect cascades.
    """
    insp_id = f"INSP-{uuid.uuid4().hex[:8].upper()}"
    log_entry = InspectionLog(
        inspection_id=insp_id,
        line_id=payload.line_id,
        is_defective=payload.is_defective,
        defect_classes=payload.defect_classes,
        confidence_scores=payload.confidence_scores,
        bounding_boxes=payload.bounding_boxes,
        inference_time_ms=payload.inference_time_ms,
        timestamp=get_current_utc()
    )
    db.add(log_entry)
    db.commit()

    # Trigger Evaluation: Fetch recent inspections (up to 30 for sliding window)
    recent_logs = db.query(InspectionLog)\
        .filter(InspectionLog.line_id == payload.line_id)\
        .order_by(desc(InspectionLog.timestamp))\
        .limit(30)\
        .all()

    incident_triggered = False
    created_ticket_id = None

    # Check if an unresolved ticket already exists
    existing_pending = db.query(MESTicket)\
        .filter(MESTicket.line_id == payload.line_id, MESTicket.status == "PENDING_APPROVAL")\
        .first()

    if not existing_pending and recent_logs:
        now_utc = get_current_utc()
        ticket_id = f"TICK-{now_utc.strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"

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
            agent_result = incident_agent.run(
                defect_class=defects_found,
                consecutive_count=3,
                line_id=payload.line_id
            )
            ticket = MESTicket(
                ticket_id=ticket_id,
                line_id=payload.line_id,
                severity="CRITICAL" if defects_found in ["short_circuit", "short"] else "MEDIUM",
                trigger_reason=f"Phát hiện 3 sản phẩm liên tiếp có lỗi '{defects_found}' trên chuyền {payload.line_id}.",
                root_cause_analysis=agent_result.get("rca_analysis", ""),
                recommended_sop=", ".join(agent_result.get("sop_citations", [])) or "SOP-SMT-001",
                action_type=agent_result.get("proposed_action", "HALT_LINE"),
                status="PENDING_APPROVAL",
                created_at=now_utc
            )
            db.add(ticket)
            db.commit()
            incident_triggered = True
            created_ticket_id = ticket_id

        elif drift_triggered:
            from collections import Counter
            all_def_types = []
            for item in recent_logs:
                if item.is_defective and item.defect_classes:
                    all_def_types.extend(item.defect_classes)
            primary_defect = Counter(all_def_types).most_common(1)[0][0] if all_def_types else "defect"

            agent_result = incident_agent.run(
                defect_class=primary_defect,
                consecutive_count=window_defects,
                line_id=payload.line_id
            )
            ticket = MESTicket(
                ticket_id=ticket_id,
                line_id=payload.line_id,
                severity="CRITICAL" if defect_ratio > 0.30 else "MEDIUM",
                trigger_reason=f"Trượt ngưỡng tỷ lệ lỗi (Yield Rate Drift): {defect_ratio*100:.1f}% lỗi ({window_defects}/{window_len}) trong cửa sổ trượt (ngưỡng cho phép: 15%). Lỗi xuất hiện nhiều nhất: '{primary_defect}'.",
                root_cause_analysis=agent_result.get("rca_analysis", ""),
                recommended_sop=", ".join(agent_result.get("sop_citations", [])) or "SOP-SMT-001",
                action_type=agent_result.get("proposed_action", "ROUTE_REWORK"),
                status="PENDING_APPROVAL",
                created_at=now_utc
            )
            db.add(ticket)
            db.commit()
            incident_triggered = True
            created_ticket_id = ticket_id

    return InspectionResponse(
        inspection_id=insp_id,
        status="RECORDED",
        incident_triggered=incident_triggered,
        ticket_id=created_ticket_id
    )

@app.post("/api/v1/mes/action")
def resolve_ticket(payload: ActionApprovalRequest, db: Session = Depends(get_db)):
    """
    Executes supervisor approval or rejection of an automated MES action.
    """
    ticket = db.query(MESTicket).filter(MESTicket.ticket_id == payload.ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    line = db.query(ProductionLine).filter(ProductionLine.line_id == ticket.line_id).first()
    now_utc = get_current_utc()

    if payload.action.upper() == "APPROVE":
        ticket.status = "APPROVED"
        ticket.approved_by = payload.approved_by
        ticket.resolved_at = now_utc
        if line and ticket.action_type == "HALT_LINE":
            line.status = "HALTED"
        db.commit()
        return {
            "status": "EXECUTED",
            "message": f"Dây chuyền {line.line_id if line else ''} đã được chuyển sang trạng thái HALTED an toàn.",
            "ticket_id": ticket.ticket_id
        }
    else:
        ticket.status = "REJECTED"
        ticket.approved_by = payload.approved_by
        ticket.resolved_at = now_utc
        db.commit()
        return {
            "status": "DISMISSED",
            "message": f"Lệnh can thiệp cho ticket {ticket.ticket_id} đã bị từ chối bởi Quản đốc.",
            "ticket_id": ticket.ticket_id
        }

@app.get("/api/v1/lines/{line_id}/metrics", response_model=LineMetricsResponse)
def get_line_metrics(line_id: str, db: Session = Depends(get_db)):
    """Retrieves operational metrics for a production line."""
    line = db.query(ProductionLine).filter(ProductionLine.line_id == line_id).first()
    if not line:
        raise HTTPException(status_code=404, detail="Line not found")

    logs = db.query(InspectionLog).filter(InspectionLog.line_id == line_id).all()
    total = len(logs)
    defects = sum(1 for log in logs if log.is_defective)
    yield_rate = ((total - defects) / total * 100.0) if total > 0 else 100.0
    avg_latency = (sum(log.inference_time_ms for log in logs) / total) if total > 0 else 0.0

    return LineMetricsResponse(
        line_id=line.line_id,
        name=line.name,
        status=line.status,
        total_inspected=total,
        defects_count=defects,
        yield_rate=round(yield_rate, 2),
        avg_inference_ms=round(avg_latency, 2)
    )

@app.get("/api/v1/tickets/recent")
def get_recent_tickets(db: Session = Depends(get_db)):
    """Fetches recently triggered tickets."""
    return db.query(MESTicket).order_by(desc(MESTicket.created_at)).limit(10).all()
