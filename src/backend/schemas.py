import datetime
from typing import List, Optional, Any
from pydantic import BaseModel, Field

class InspectionCreate(BaseModel):
    line_id: str = "SMT-LINE-01"
    is_defective: bool
    defect_classes: List[str] = []
    confidence_scores: List[float] = []
    bounding_boxes: List[Any] = []
    inference_time_ms: float

class InspectionResponse(BaseModel):
    inspection_id: str
    status: str
    incident_triggered: bool
    ticket_id: Optional[str] = None

class ActionApprovalRequest(BaseModel):
    ticket_id: str
    action: str = Field(..., description="APPROVE or REJECT")
    approved_by: str = "supervisor_on_duty"
    idempotency_key: Optional[str] = Field(None, max_length=128)

class LineMetricsResponse(BaseModel):
    line_id: str
    name: str
    status: str
    total_inspected: int
    defects_count: int
    yield_rate: float
    avg_inference_ms: float
