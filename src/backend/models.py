import datetime
from sqlalchemy import Column, String, Float, Boolean, DateTime, Text, JSON
from .database import Base

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)

class ProductionLine(Base):
    """Represents a factory SMT production line."""
    __tablename__ = "production_lines"

    line_id = Column(String(50), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    status = Column(String(20), default="RUNNING") # RUNNING, WARNING, HALTED
    current_product = Column(String(100), nullable=False)
    target_yield_rate = Column(Float, default=95.0)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

class InspectionLog(Base):
    """Telemetry record for each inspected item passed under the camera."""
    __tablename__ = "inspections"

    inspection_id = Column(String(64), primary_key=True, index=True)
    line_id = Column(String(50), index=True)
    timestamp = Column(DateTime, default=utc_now, index=True)
    image_filename = Column(String(255), nullable=True)
    is_defective = Column(Boolean, nullable=False, index=True)
    defect_classes = Column(JSON, default=list)        # ['short_circuit', 'mouse_bite']
    confidence_scores = Column(JSON, default=list)     # [0.92, 0.85]
    bounding_boxes = Column(JSON, default=list)        # [[x1, y1, x2, y2], ...]
    inference_time_ms = Column(Float, nullable=False)

class MESTicket(Base):
    """Maintenance and line dispatch action tickets created by Agentic AI."""
    __tablename__ = "mes_tickets"

    ticket_id = Column(String(64), primary_key=True, index=True)
    line_id = Column(String(50), index=True)
    severity = Column(String(20), nullable=False)      # LOW, MEDIUM, CRITICAL
    trigger_reason = Column(Text, nullable=False)
    root_cause_analysis = Column(Text, nullable=False)
    recommended_sop = Column(String(100), nullable=False)
    action_type = Column(String(50), nullable=False)   # HALT_LINE, ROUTE_REWORK, CALIBRATE
    status = Column(String(20), default="PENDING_APPROVAL") # PENDING_APPROVAL, APPROVED, REJECTED, EXECUTED
    approved_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=utc_now)
    resolved_at = Column(DateTime, nullable=True)
