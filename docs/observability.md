# ApexInspect AI — Factory Observability Guide

> Mục tiêu: cho phép operator và engineer monitor sức khỏe dây chuyền, phát hiện sớm khi line có vấn đề, và audit mọi actiontight.

## 1. Metrics có sẵn

### 1.1 Per-frame Inference Latency
- **Nơi đo**: `PCBDefectDetector.infer()` → `inference_time_ms` trả về, ghi vào `inspection_log.inference_time_ms`.
- **Tại sao quan trọng**: latency tăng đột biến có thể từ hardware issue, model chậm, hoặc input hỏng.
- **Access**: `GET /api/v1/lines/{line_id}/metrics` → `avg_inference_ms`.
- **Alert threshold**: nếu `avg_inference_ms > 50ms` trong 1 phút → warning.

### 1.2 Yield Rate
- **Công thức**: `(total - defects) / total * 100%`.
- **Access**: `GET /api/v1/lines/{line_id}/metrics` → `yield_rate`.
- **Alert threshold**: nếu `yield_rate < 85%` → incident trigger (consistent với `QAI` graph).

### 1.3 Defect Pareto
- Phân loại defect theo class (missing_hole, mouse_bite, open_circuit, short_circuit, spur, spurious_copper).
- Dùng cho quality improvement meeting — biết loại lỗi nào chiếm đa số.

### 1.4 Audit Trail (Immutable)
- Mọi action ghi vào `audit_logs`:
  - `TRIGGER_CONSECUTIVE_DEFECTS` — khi ≥3 defect liên tiếp.
  - `TRIGGER_YIELD_DRIFT` — khi yield rate < 85%.
  - `APPROVE_ACTION` — supervisor approve.
  - `REJECT_ACTION` — supervisor reject.
  - `RESUME_LINE` — line resume.
- **Immutable**: không có delete/update trên audit_logs (theo design).
- **Access**: `GET /api/v1/tickets/recent` + query trực tiếp bảng.

### 1.5 HITL Checkpoint State
- LangGraph `MemorySaver` giữ state của interrupted workflow.
- Operator có thể xem incident đang pending approval trên Streamlit UI.
- Resume sau khi approve → state transition `PENDING → APPROVED/REJECTED`.

### 1.6 PLC Actuation Status
- `PLCBridge` trả về status: `SIMULATED` (fallback) hoặc `DISPATCHED` (real).
- Nếu `SIMULATED` trong production — cần investigate vì PLC có thể offline.
- Telemetry: `conveyor_running`, `tower_light` (Red/Yellow/Green), `halt_triggered`.

## 2. Metrics nào giảm sát khi line dừng?

| Metric | Normal | Khi line dừng | Tại sao |
|--------|--------|---------------|---------|
| Yield rate | ≥95% | ↓↓ (có thể 0% nếu line halt) | Không sản phẩm chạy qua |
| Consecutive defect count | 0-2 | ≥3 (trigger dừng) | Tích lũy defect trước khi halt |
| Inference latency | ≤35ms target | ↑ (có thể dari hardware) | Nếu dừng do hardware issue |
| Conveyor running | True | False | Halt line → conveyor stop |
| Tower light | GREEN | RED | Halt → red light |

## 3. Dashboard Streamlit

Mở `http://localhost:8501` (hoặc port Kiểm tra .env) → các tab:
- **Live Feed**: camera stream + bounding box + latency HUD.
- **Incident Center**: danh sách ticket đang pending, approve/reject.
- **Yield Analytics**: biểu đồ yield rate theo thời gian, defect pareto.

## 4. Audit Log Example

```sql
SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 10;
```

Output expected:
```
log_id               | timestamp           | operator_id      | action               | line_id      | ticket_id            | details
------------------------------------------------------------------------------------------------------------------------------------------------------------------
audit-20260917-001   | 2026-09-17 01:00:00 | supervisor_tan  | APPROVE_ACTION       | SMT-LINE-01  | TICK-20260917-001    | Approved HALT_LINE
audit-20260917-002   | 2026-09-17 00:55:00 | system          | TRIGGER_CONSECUTIVE  | SMT-LINE-01  | TICK-20260917-001    | 3 consecutive short_circuit
```

## 5. Log-level Debug

- OpenCV log: `OPENCV_LOG_LEVEL=ERROR` (mặc định trong conftest).
- ONNX Runtime: không có log enable sẵn — có thể bật `ORT_LOG_LEVEL` nếu cần.
- App log: `print()` statement trong code — nên chuyển sang `logging` module cho production.

## 6. Future: Metrics Export

- Hiện tại metrics chỉ expose qua API + Streamlit.
- Nếu cần integration với external monitoring (Prometheus, Grafana), có thể add `/metrics` endpoint expose Prometheus format.
- PLC telemetry (holding registers) có thể expose qua Modbus — nhưng cần PLC support.