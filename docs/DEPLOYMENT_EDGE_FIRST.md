# DEPLOYMENT CHỐT — Edge-First (cập nhật plan 2026-09-18 Incident Actuation Safety)

## PLC: simulation vs hardware

| `APEX_PLC_MODE` | Hành vi |
|---|---|
| `simulation` (default) | Không connect, không write ra bất kỳ thiết bị nào. Mọi action trả `mode=SIMULATION`, `status=SIMULATED`. |
| `hardware` | Kết nối Modbus TCP thật (`PLC_HOST`/`PLC_PORT`). Mỗi write response được kiểm tra; lỗi trả `status=FAILED` + `error` có cấu trúc — **không bao giờ** bị đổi thành "success" hay simulation. |

- PLC chưa connect ở hardware mode → `PLC_UNAVAILABLE`, ticket vào
  `ACTUATION_FAILED` (HTTP 502 từ gateway), line state không bị đổi vội.
- Mọi test dùng injected recording bridge; không test nào chạm PLC thật.

## Ticket idempotency + approval

- `POST /api/v1/mes/action` reserve ticket bằng conditional update
  `PENDING_APPROVAL → RESOLVING`; approve trùng lặp → HTTP 409, **tối đa một**
  lệnh PLC cho mỗi ticket.
- Ticket giữ `resolution_key`, `plc_status`, `plc_result_json` làm bằng chứng
  bền vững. Terminal states: `EXECUTED | ACTUATION_FAILED | REJECTED`.
- Ticket tạo qua agent HITL có `thread_id` → service được dùng simulation
  bridge mặc định (không bao giờ hardware); gọi service trực tiếp không có
  bridge → fail-closed `ACTUATION_FAILED`.

## Approval state bền vững (checkpointer)

- `src/agent/checkpointer.py` — `IncidentCheckpointer` (SQLite, JSON-only):
  mỗi ticket/thread một hàng, không serialize object chạy được. Agent instance
  mới resume đúng incident sau restart.
- Approval chỉ resume đúng graph sau khi ticket reservation thành công; graph
  trả về PLC outcome thật đã persist (không còn success string).

## Phục hồi thủ công `ACTUATION_FAILED`

1. Kiểm tra `ticket.plc_result_json` (lý do: `PLC_UNAVAILABLE` /
   `PLC_WRITE_REJECTED` / exception string).
2. Sửa phần cứng/kết nối; xác nhận `APEX_PLC_MODE` đúng ý đồ vận hành.
3. Tạo ticket mới (ticket cũ là terminal — không replay lệnh đã fail, tránh
   double-actuation). Audit trail giữ nguyên bằng chứng.

## Nền tảng / lưu ý OpenCV

- Suite vision (`tests/test_ingest.py`, `test_patching.py`,
  `test_prepare_dataset.py`, `test_quality.py`, `test_stream.py`,
  `test_vision.py`, `test_sample_gallery.py`) cần `opencv-python` — trên một số
  nền tảng (ví dụ macOS Python 3.14 hiện tại) wheel chưa khả dụng. Đây là yêu
  cầu nền tảng **được ghi nhận thẳng**, không bị che: các test an toàn còn lại
  (57 passed, 2 skipped khi bỏ 7 module cv2) vẫn là cổng CI bắt buộc; CI Linux
  cài `requirements.txt` đầy đủ nên chạy trọn bộ.

---


# DEPLOYMENT CHỐT: Edge-First (Offline On-Premise)

Deployment duy nhất: **EDGE-FIRST** — YOLOv8 ONNX chạy 100% tại Edge PC công nghiệp
(target ≤15ms/frame). RCA ưu tiên Local/Deterministic (Heuristic rule-based + BM25
inference thuần, `APEX_LLM_MODE=deterministic`). Cloud (Groq/Neon/Render) chỉ là
Control Plane nhận báo cáo bất đồng bộ — không nằm luồng chính, offline 100%.
Training duy nhất trên Colab: Vision finetune YOLOv8n (`colab/train_yolo_T4.ipynb`).

```text
[Colab T4: finetune YOLOv8n → ONNX] ──weights──▶ [Edge PC: Camera → ONNX ≤15ms → RCA local → HITL → PLC]
                                                        │ (async reports only, offline-safe)
                                                        ▼
                                                  [Cloud Control Plane: Neon/Render/Groq]
```

Verify: `grep -rn "Edge-First\|EDGE" README.md docs/ plans/ tasks/ .env.example`
+ `APEX_LLM_MODE=deterministic` trong `.env.example`, test suite offline xanh.
