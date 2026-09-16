import os
import uuid
import datetime
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
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

@app.get("/health")
def health_check():
    """Liveness probe for Docker HEALTHCHECK (Space API) and post-deploy smoke tests."""
    return {"status": "ok", "service": "apexinspect-gateway", "version": app.version}

from .service import InspectionService
from .models import ProductionLine, InspectionLog, MESTicket, AuditLog

# Security: Industrial API Key Header
API_KEY_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=False)

def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> Optional[str]:
    """Validates X-API-KEY for critical industrial action dispatch."""
    expected = os.getenv("API_KEY", "dev-factory-key-secret").strip()
    if not expected or expected.lower() in ("none", "development", "disabled"):
        return api_key
    if api_key and api_key == expected:
        return api_key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required 'X-API-KEY' authentication header."
        )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid 'X-API-KEY' credential."
    )

# Shared agent and industrial PLC bridge instances
incident_agent = QualityIncidentAgent()

try:
    from ..industrial.plc_bridge import PLCBridge
    plc_bridge = PLCBridge()
except Exception:
    plc_bridge = None

def get_current_utc():
    return datetime.datetime.now(datetime.timezone.utc)

@app.post("/api/v1/inspections", response_model=InspectionResponse)
def record_inspection(payload: InspectionCreate, db: Session = Depends(get_db)):
    """
    Ingests defect telemetry from edge camera stream via InspectionService,
    checks incident triggers, and invokes LangGraph Quality Incident Agent upon cascades.
    """
    log_entry, incident_triggered, created_ticket_id = InspectionService.record_telemetry(
        db=db,
        payload=payload,
        agent=incident_agent
    )

    return InspectionResponse(
        inspection_id=log_entry.inspection_id,
        status="RECORDED",
        incident_triggered=incident_triggered,
        ticket_id=created_ticket_id
    )

@app.post("/api/v1/mes/action", dependencies=[Depends(verify_api_key)])
def resolve_ticket(payload: ActionApprovalRequest, db: Session = Depends(get_db)):
    """
    Executes supervisor approval or rejection of an automated MES action with LangGraph resume.
    """
    ticket = db.query(MESTicket).filter(MESTicket.ticket_id == payload.ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    result = InspectionService.resolve_ticket(
        db=db,
        ticket_id=payload.ticket_id,
        action=payload.action,
        approved_by=payload.approved_by or "supervisor_on_duty",
        agent=incident_agent,
        plc_bridge=plc_bridge
    )

    if result.get("status") == "ERROR":
        raise HTTPException(status_code=400, detail=result.get("message"))

    return result

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
