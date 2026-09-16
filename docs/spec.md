# Product Specification: ApexInspect AI
**Autonomous Industrial Vision & Defect Resolution Agent System**

## 1. Executive Summary & Problem Statement
Trong môi trường sản xuất công nghiệp và chế tạo điện tử (SMT / PCBA assembly), việc kiểm soát chất lượng sản phẩm (QA/QC) truyền thống đối mặt với 2 rào cản lớn:
1. **Kiểm tra ngoại quan thủ công (Manual Optical Inspection)**: Tốn nhân lực, độ tập trung giảm theo thời gian ca làm việc, tỷ lệ bỏ sót các vi lỗi như chân hàn chập (short circuit), mất lỗ linh kiện (missing hole) hoặc vết xước bề mặt là rất cao.
2. **Hệ thống Vision truyền thống bị phân mảnh (Siloed Machine Vision)**: Các camera công nghiệp truyền thống chỉ dừng lại ở việc báo cờ đỏ Pass/Fail. Chúng thiếu một **bộ não điều phối thông minh** để:
   - Tự động tra cứu tài liệu quy trình vận hành tiêu chuẩn (SOP).
   - Đưa ra khuyến nghị chẩn đoán nguyên nhân gốc rễ (Root Cause Analysis - RCA).
   - Tương tác hai chiều với hệ thống điều hành sản xuất (**MES** - Manufacturing Execution System) để dừng chuyền hoặc điều hướng sản phẩm lỗi sang trạm Rework.

**ApexInspect AI** được thiết kế như một hệ sinh thái khép kín:
$$\text{Camera Stream (OpenCV)} \longrightarrow \text{ONNX Defect Detection} \longrightarrow \text{Telemetry Ingestion} \longrightarrow \text{LangGraph Agent + RAG SOP} \longrightarrow \text{HITL Approval} \longrightarrow \text{MES Sync}$$

---

## 2. Target Users & Operating Scenarios
- **Quản đốc chuyền / Kỹ sư QA (Line Supervisor)**:
  - Giám sát luồng video camera thời gian thực, nắm bắt tỷ lệ lỗi (Yield Rate %) theo từng ca sản xuất.
  - Nhận cảnh báo sự cố tức thời khi hệ thống phát hiện chuỗi lỗi nghiêm trọng lặp lại.
  - Phê duyệt hành động can thiệp (Human-in-the-Loop) như dừng chuyền hoặc điều hướng trạm rework.
- **Kỹ sư bảo trì thiết bị (Maintenance Engineer)**:
  - Nhận ngay gợi ý xử lý sự cố đính kèm trích dẫn điều khoản SOP chính xác, thông số hiệu chuẩn máy mà không cần tốn thời gian lật tìm tài liệu giấy.
- **Giám đốc nhà máy (Plant Manager)**:
  - Xem báo cáo tổng hợp tỷ lệ lỗi, biểu đồ Pareto phân bổ nguyên nhân lỗi phục vụ họp giao ban sản xuất.

---

## 3. Core Functional Requirements (FR)

### FR-1: High-Speed Edge Defect Detection (Computer Vision)
- **Input**: Luồng video hoặc chuỗi hình ảnh công nghiệp (640x640) mô phỏng camera trên băng tải SMT.
- **Model**: YOLOv8 Nano fine-tuned cho bài toán PCB Surface Defect, xuất sang định dạng **ONNX (Open Neural Network Exchange)**.
- **Processing**: Sử dụng OpenCV để tiền xử lý (cắt ROI, cân bằng sáng CLAHE, khử nhiễu) và `onnxruntime` để suy luận.
- **Output**: Bounding boxes, phân loại lỗi (`short_circuit`, `missing_hole`, `open_circuit`, `mouse_bite`, `spur`), điểm tin cậy (confidence score $\ge 0.50$) và độ trễ suy luận (Inference Latency $\le 35\text{ ms}$).

### FR-2: Telemetry Ingestion & Incident Threshold Engine
- Mọi lượt soi từ camera đều được ghi nhận (Pass / Defect) vào cơ sở dữ liệu.
- Tự động kích hoạt AI Agent khi thỏa mãn một trong hai điều kiện ngưỡng:
  - **Consecutive Defect Trigger**: Phát hiện $\ge 3$ sản phẩm liên tiếp cùng dính lỗi nghiêm trọng (ví dụ: `short_circuit`).
  - **Yield Rate Drift Trigger**: Tỷ lệ lỗi trong cửa sổ trượt 30 sản phẩm gần nhất vượt quá $15\%$.

### FR-3: SOP Knowledge Retrieval & Root Cause Analysis (Agentic RAG)
- Lưu trữ kho quy trình thao tác chuẩn (SOP) và cẩm nang bảo trì thiết bị dán bề mặt / lò hàn hồi lưu.
- Trích xuất thông tin qua cơ chế Hybrid RAG (Dense vector search + BM25 keyword matching).
- Áp dụng nguyên tắc **Zero Hallucination**: Trích dẫn minh bạch mã quy trình `[SOP-SMT-xxx]`. Nếu tài liệu không đề cập nguyên nhân, Agent phải từ chối phỏng đoán.

### FR-4: Human-in-the-Loop (HITL) Action Execution
- AI Agent **không tự ý ra lệnh dừng chuyền** mà chỉ lập dự thảo hành động (Action Proposal).
- Sử dụng cơ chế checkpoint `interrupt` của LangGraph để tạm dừng workflow và chờ phê duyệt.
- Quản đốc bấm `[Phê duyệt]` hoặc `[Từ chối]` trên giao diện điều hành. Lệnh chỉ được thi hành khi có xác nhận của con người.

### FR-5: Simulated MES / ERP Synchronization
- Giả lập hệ thống điều hành sản xuất MES:
  - Cập nhật trạng thái dây chuyền (`RUNNING` $\leftrightarrow$ `HALTED`).
  - Tự động tạo `Maintenance Ticket` với đầy đủ mã lỗi, ảnh chụp vi phạm và khuyến nghị SOP.

### FR-6: Industrial OT Actuation & Modbus TCP PLC Bridge
- Kết nối hai chiều với bộ điều khiển PLC công nghiệp (Siemens S7-1200 / Omron NX1P2 / Beckhoff) qua giao thức chuẩn **Modbus TCP**:
  - Ghi Coils kích hoạt ngắt nguồn băng tải khẩn cấp (`COIL_HALT_LINE`).
  - Kích hoạt pít-tông khí nén gạt sản phẩm lỗi sang khay rework (`COIL_REWORK_DIVERT`).
  - Điều khiển đèn tháp tín hiệu Andon 3 màu (`TOWER_LIGHT_RED`, `TOWER_LIGHT_YELLOW`, `TOWER_LIGHT_GREEN`).
  - Đồng bộ sản lượng và tỷ lệ lỗi vào các thanh ghi Holding Registers phục vụ hệ thống SCADA.
  - Tích hợp máy chủ giả lập ảo (`VirtualModbusServer`) cho môi trường dev/test không cần phần cứng vật lý.

### FR-7: High-Resolution Sliced Inference (SAHI Patching)
- Cung cấp module `HighResPatchInferencer` chia nhỏ ảnh độ phân giải siêu cao (2048x2048, 4096x4096) thành các lát cắt gối đầu (sliding patches với overlap ratio 20%).
- Khử trùng lặp đa vùng bằng Non-Maximum Suppression (NMS) toàn cục, chống bỏ sót các lỗi kích thước hiển vi (chân hàn chập sợi tóc, khuyết đồng mouse bite 10-15px).

### FR-8: Resilient Threaded Industrial Camera Stream Ingestion
- Bộ thu hình đa luồng (`IndustrialCameraStream`) tương thích luồng RTSP từ camera công nghiệp GigE/USB3 Vision.
- Cơ chế xả đệm thông minh (zero-lag latest frame drop policy) đảm bảo độ trễ thời gian thực.
- Tự động kiểm tra chất lượng hình ảnh (độ mờ nét Laplacian variance, thiếu sáng, quá sáng) và tự động kết nối lại khi đứt cáp mạng.

### FR-9: IPC-A-610 Class 3 & SMT Standard Operating Procedures
- Bổ sung bộ tài liệu quy chuẩn kỹ thuật quốc tế:
  - `SOP-SMT-004`: Tiêu chuẩn nghiệm thu mối hàn theo IPC-A-610 Class 3.
  - `SOP-SMT-005`: Kiểm soát và hiệu chuẩn lệch đường cong nhiệt độ lò hàn hồi lưu (Reflow Profile Drift / PWI / TAL).
  - `SOP-SMT-006`: Quy trình vệ sinh siêu âm và bảo trì đầu hút chân không máy gắp linh kiện (SMD Nozzle Maintenance).

---

## 4. Free Infrastructure Deployment Architecture (Zero-Cost Stack)

Toàn bộ hệ thống được tối ưu hóa để triển khai **hoàn toàn miễn phí** với độ sẵn sàng cao:

| Thành phần | Dịch vụ Miễn phí | Thông số kỹ thuật | Vai trò trong hệ thống |
| :--- | :--- | :--- | :--- |
| **Model Serving & Web UI** | **Hugging Face Spaces** | 2 vCPU, 16 GB RAM, Docker SDK | Chạy trọn gói Streamlit Dashboard, OpenCV, ONNX Runtime CPU và FastAPI Gateway. |
| **Managed Database** | **Neon PostgreSQL** | 0.5 GB Storage, Serverless Postgres | Lưu trữ bảng dữ liệu kiểm định, trạng thái dây chuyền, audit log và ticket MES. |
| **LLM Inference** | **Groq API** | Llama 3.3 70B, 30 req/min free | Bộ não Agent phân tích nguyên nhân sự cố và tổng hợp SOP tốc độ ~300 tokens/s. |
| **Model Weight Storage** | **GitHub Releases / Git LFS** | Băng thông miễn phí | Lưu file trọng số `yolov8n_pcb_defect.onnx` (<15 MB). |

---

## 5. Non-Functional Requirements (NFR)
1. **Performance**: Thông lượng xử lý hình ảnh trên CPU $\ge 25\text{ FPS}$ (phù hợp với tốc độ băng tải sản xuất tiêu chuẩn).
2. **Reliability & Fallback**: Nếu mất kết nối Neon PostgreSQL, hệ thống tự động chuyển sang lưu tạm thời trên SQLite cục bộ (`factory.db`). Nếu mất kết nối Groq API, Agent chuyển sang chế độ template quy trình offline.
3. **Auditability & Traceability**: Bảng kiểm toán `audit_logs` lưu trữ mọi hành động can thiệp, phân luồng sự cố, IP nguồn và chữ ký số Quản đốc ca.
4. **Security**: Bảo vệ cổng điều khiển tác vụ sản xuất MES với tiêu chuẩn xác thực mã khóa `X-API-KEY`.
5. **Testing**: Độ bao phủ kiểm thử toàn diện 100% với 61 test cases bao quát toàn bộ Vision, Agent HITL, Database, Modbus PLC, Patching và Camera Stream.
