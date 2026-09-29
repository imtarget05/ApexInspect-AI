import os
import re
import json
import time
import uuid
import datetime
import hmac
import secrets
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi import Request
from sqlalchemy.orm import Session
from sqlalchemy import desc

from .database import get_db, init_db
from .models import ProductionLine, InspectionLog, MESTicket
from .schemas import InspectionCreate, InspectionResponse, ActionApprovalRequest, LineMetricsResponse

#: Correlation ID do client gửi lên chỉ được chấp nhận nhiệt liệt nếu khớp
#: mẫu này. Chấp nhận mọi chuỗi tùy ý nghĩa là log của ta có thể bị chèn
#: newline giả mạo dòng log khác — log injection.
_SAFE_CID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
from ..agent.graph import QualityIncidentAgent

_DEVELOPMENT_ENVIRONMENTS = {"", "dev", "development", "local", "test"}
_ephemeral_api_key: Optional[str] = None


def _is_development() -> bool:
    """True unless APP_ENV/ENVIRONMENT names a deployed environment.

    There was no environment marker in this service before; API_KEY now needs
    one to fail closed without breaking local work.
    """
    raw = os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or "development"
    return raw.strip().lower() in _DEVELOPMENT_ENVIRONMENTS


def _expected_api_key() -> str:
    """The configured API key, or a hard startup failure outside development.

    There is deliberately no committed default. A deployment that forgets
    API_KEY must fail to start, not run on a credential that is published in
    this repository. Locally, an unset API_KEY mints a random per-process key
    so development still works without shipping a well-known secret.
    """
    global _ephemeral_api_key
    raw = os.getenv("API_KEY", "").strip()
    if raw:
        return raw
    if not _is_development():
        raise RuntimeError(
            "API_KEY must be set when APP_ENV is not a development environment. "
            "Generate one with `python -c \"import secrets; print(secrets.token_urlsafe(32))\"` "
            "and set it in the deployment environment. There is no default: a "
            "missing key must not silently fall back to a published value."
        )
    if _ephemeral_api_key is None:
        _ephemeral_api_key = secrets.token_urlsafe(32)
        print(
            "[security] API_KEY is not set and APP_ENV is a development "
            f"environment. Using a random ephemeral key for this process only: "
            f"{_ephemeral_api_key}"
        )
    return _ephemeral_api_key


def _resolve_cors_origins() -> list:
    """Explicit origin allowlist. Never a wildcard alongside credentials.

    `allow_origins=["*"]` together with `allow_credentials=True` is rejected by
    browsers and, where honoured, lets any site read authenticated responses.
    """
    raw = os.getenv("CORS_ORIGINS", "")
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    if "*" in origins:
        raise RuntimeError(
            "CORS_ORIGINS must list explicit origins; the wildcard '*' cannot be "
            "combined with credentialed requests. List each trusted UI origin, "
            "comma-separated."
        )
    return origins


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail closed before serving anything if the deployment has no API_KEY.
    _expected_api_key()
    init_db()
    yield

app = FastAPI(
    title="ApexInspect AI Gateway",
    description="Industrial Telemetry Ingestion, Automated Defect Resolution & MES Synchronization",
    version="1.0.0",
    lifespan=lifespan
)

_CORS_ORIGINS = _resolve_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-KEY", "X-Correlation-ID"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """
    Gắn correlation ID cho mọi request để điều tra được một luồng.

    Vì sao cần: khi một inspection bị chặn giữa PLC, MES và DB, log của
    từng thành phần nằm ở process khác nhau và chỉ liên kết được bằng một
    mã chung. Không có mã đó, điều tra phải đoán theo thời gian — và
    điều tra theo thời gian thì chậm và sai.

    ID đến từ client (`X-Correlation-ID`) được giữ nguyên khi hợp lệ, để
    trace xuyên qua các service khác. ID do ta sinh thì dùng UUID v4.

    ID sinh ra KHÔNG được ghi vào DB theo mặc định: nó do người gọi kiểm
    soát, và lưu nó mở đường cho log injection. Ở đây nó chỉ đi kèm log
    của request, do chính process sinh ra.
    """
    inbound = request.headers.get("X-Correlation-ID", "")
    cid = inbound[:64] if _SAFE_CID.match(inbound) else uuid.uuid4().hex
    request.state.correlation_id = cid
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Correlation-ID"] = cid
    # Log có cấu trúc: một dòng JSON mỗi request, thay vì dòng access log
    # không có ngữ cảnh. Đây là nơi người ta grep khi điều tra.
    print(json.dumps({
        # File này import `datetime` là MODULE, không phải class. Viết
        # `datetime.now(...)` sẽ ném AttributeError và làm mọi request 500.
        # `timezone` thì lấy từ module datetime.
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "level": "INFO",
        "service": "apexinspect-gateway",
        "event": "http_request",
        "correlation_id": cid,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": round(elapsed_ms, 2),
    }), flush=True)
    return response

@app.get("/health")
def health_check():
    """Liveness probe for Docker HEALTHCHECK (Space API) and post-deploy smoke tests."""
    return {"status": "ok", "service": "apexinspect-gateway", "version": app.version}


def _check_database_configured() -> None:
    """Non-mutating config probe: a supported DATABASE_URL or a resolvable
    SQLite fallback must exist. Opens no connections, writes nothing."""
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        scheme = url.split("://", 1)[0].lower()
        if scheme not in ("sqlite", "postgresql", "postgres"):
            raise RuntimeError(f"unsupported DATABASE_URL scheme: {scheme}")
        return
    fallback = os.getenv("SQLITE_PATH", "./factory.db")
    parent = os.path.dirname(os.path.abspath(fallback)) or "."
    if not os.path.isdir(parent):
        raise RuntimeError(f"sqlite fallback directory missing: {parent}")


def _check_model_bundle_present() -> None:
    """Non-mutating probe: model bundle file must exist. Loads nothing."""
    model_path = os.getenv("MODEL_PATH", "models/yolov8n_pcb_defect.onnx")
    if not os.path.isfile(model_path):
        raise RuntimeError(f"model bundle missing: {model_path}")


@app.get("/health/live")
def health_live():
    """Liveness: process can serve. Requires no external dependencies."""
    return {"status": "ok", "service": "apexinspect-gateway", "version": app.version}


@app.get("/health/ready")
def health_ready():
    """Readiness: config valid + required dependencies reachable. No writes."""
    checks = {"config": "ok"}
    try:
        _check_database_configured()
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"not-ready: {exc}"
    try:
        _check_model_bundle_present()
        checks["model_bundle"] = "ok"
    except Exception as exc:
        checks["model_bundle"] = f"not-ready: {exc}"
    ready = all(value == "ok" for value in checks.values())
    return {"status": "ready" if ready else "not-ready", "checks": checks}

from .service import InspectionService
from .models import ProductionLine, InspectionLog, MESTicket, AuditLog

# Security: Industrial API Key Header
API_KEY_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=False)

def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> Optional[str]:
    """Validates X-API-KEY for protected industrial routes.

    No default key, and no "disabled" sentinel: an unset API_KEY outside
    development raises rather than opening the gate. Comparison is
    constant-time so the key cannot be recovered by timing the endpoint.
    """
    expected = _expected_api_key()
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required 'X-API-KEY' authentication header."
        )
    if hmac.compare_digest(api_key.encode("utf-8"), expected.encode("utf-8")):
        return api_key
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

@app.post("/api/v1/inspections", response_model=InspectionResponse, dependencies=[Depends(verify_api_key)])
def record_inspection(payload: InspectionCreate, db: Session = Depends(get_db)):
    """Ingests defect telemetry from the edge camera stream.

    API-key protected, same as POST /api/v1/mes/action. This is an edge-camera
    write path, which does not make it public: the row it writes feeds the
    3-consecutive-defect trigger, and that trigger opens a PENDING_APPROVAL MES
    ticket whose approval dispatches real PLC actuation (HALT_LINE). An
    unauthenticated caller could therefore drive the halt-line proposal path
    simply by posting three defective inspections, so the same credential
    guards it. Edge agents send the key in X-API-KEY.
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
        plc_bridge=plc_bridge,
        idempotency_key=payload.idempotency_key,
    )

    if result.get("status") == "ERROR":
        raise HTTPException(status_code=400, detail=result.get("message"))
    if result.get("status") == "CONFLICT":
        raise HTTPException(status_code=409, detail=result.get("message"))
    if result.get("status") == "ACTUATION_FAILED":
        raise HTTPException(status_code=502, detail=result.get("message"))

    return result

@app.get("/api/v1/lines/{line_id}/metrics", response_model=LineMetricsResponse, dependencies=[Depends(verify_api_key)])
def get_line_metrics(line_id: str, db: Session = Depends(get_db)):
    """Operational metrics for a production line (API-key protected).

    Yield rate and defect counts are proprietary line performance data; the
    endpoint previously exposed them to any unauthenticated caller.
    """
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

@app.get("/metrics")
def prometheus_metrics(db: Session = Depends(get_db)):
    """Prometheus text exposition (unauthenticated, aggregate-only, no PII).

    Scraped by observability/prometheus. Contains only aggregate counters:
    totals per line and status, never per-inspection proprietary detail rows.
    """
    from fastapi.responses import PlainTextResponse

    lines = db.query(ProductionLine).all()
    totals = db.query(InspectionLog).all()
    n_total = len(totals)
    n_defects = sum(1 for log in totals if log.is_defective)
    avg_ms = (sum(log.inference_time_ms for log in totals) / n_total) if n_total else 0.0

    out = [
        "# HELP apex_inspections_total Total inspections recorded.",
        "# TYPE apex_inspections_total counter",
        f"apex_inspections_total {n_total}",
        "# HELP apex_defects_total Total defective units.",
        "# TYPE apex_defects_total counter",
        f"apex_defects_total {n_defects}",
        "# HELP apex_yield_rate_ratio Yield rate (0..1).",
        "# TYPE apex_yield_rate_ratio gauge",
        f"apex_yield_rate_ratio {(((n_total - n_defects) / n_total) if n_total else 1.0):.4f}",
        "# HELP apex_avg_inference_ms Average inference latency.",
        "# TYPE apex_avg_inference_ms gauge",
        f"apex_avg_inference_ms {avg_ms:.2f}",
        "# HELP apex_line_status Line status (1=RUNNING, 0=STOPPED).",
        "# TYPE apex_line_status gauge",
    ]
    for line in lines:
        running = 1 if str(line.status).upper() in ("RUNNING", "ACTIVE", "1") else 0
        out.append(f'apex_line_status{{line="{line.line_id}",name="{line.name}"}} {running}')
    return PlainTextResponse("\n".join(out) + "\n", media_type="text/plain; version=0.0.4")


@app.get("/api/v1/tickets/recent", dependencies=[Depends(verify_api_key)])
def get_recent_tickets(db: Session = Depends(get_db)):
    """Fetches recently triggered tickets (API-key protected).

    Ticket ids, thread ids and trigger reasons are internal incident data.
    """
    return db.query(MESTicket).order_by(desc(MESTicket.created_at)).limit(10).all()
