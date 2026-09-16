# docs/adr/001-bm25-vs-vector-rag.md
# ADR-001: BM25 (Okapi) thay vì Vector Embedding cho SOP RAG

## Status
Accepted (2026-09-16)

## Context
Hệ thống cần truy xuất SOP (Standard Operating Procedure) domain SMT/PCB — IPC-A-610, reflow profile, pick & place — để hỗ trợ RCA (Root Cause Analysis) khi incident触发. Có 2 hướng:

1. **Vector embedding + similarity search** (chromadb/FAISS): phổ biến, semantic search, nhưng cần embed model, compute, và thường "distract" bởi ngữ nghĩa rộng khi các SOP cơ khí có thuật ngữ chuyên ngành rời rạc.
2. **BM25 (Okapi)**: keyword/TF-IDF-based, nhẹ, deterministic, không cần GPU/embed, với domain-specific synonym expansion có thể điều chỉnh chính xác precision.

## Decision
Chọn **BM25 (rank-bm25)** làm trọng tâm retrieval, kết hợp synonym expansion thủ công cho domain PCB/SMT.

Lý do:
- **Dataset nhỏ & cố định**: ~6 SOP files (SOP-001..006), mỗi file ngắn (1-3 pages). BM25 đủ với corpus nhỏ.
- **Không cần semantic drift**: IOP-A-610 Class 3 solder criteria, TAL/PWI, nozzle maintenance — thuật ngữ này cần exact match, không "nghĩa tương tự".
- **Offline/edge-deployable**: Không phụ thuộc external embed service, phù hợp industrial OT environment.
- **Debuggable**: Score BM25 interpretable, dễ audit khi có incident.

## Trade-offs
- Không đo được "semantic similarity" (ví dụ: "solder joint" vs "tin connection" nếu không trong synonym map).
- Phải maintain synonym expansion dict thủ công.
- Không scale tốt nếu corpus > 10k documents.

## Consequences
- Thêm `src/agent/rag.py` với `BM25Retriever` + synonym map.
- Không dùng chromadb cho SOP retrieval (chromadb vẫn có trong requirements cho mục đích khác nếu needed).
- Nếu sau này cần semantic fallback, có thể hybrid BM25 + vector làm phase 2.

---

# docs/adr/002-modbus-tcp-virtual-plc-simulator.md
# ADR-002: Modbus TCP + Virtual PLC Simulator cho Industrial OT

## Status
Accepted (2026-09-16)

## Context
Hệ thống cần giao tiếp với industrial PLC (conveyor, reject diverter, andon tower light) qua Modbus TCP — protocol phổ biến trong SMT line. Tuy nhiên:

- Phòng thí nghiệm / dev environment không có hardware PLC thật.
- Cần test tích hợp Modbus mà không phụ thuộc hardware.
- Cần simulation fallback khi chạy production mà PLC offline.

## Decision
Thêm 2 thành phần:

1. **`PLCBridge`** (`src/industrial/plc_bridge.py`): Modbus TCP client bản quyền, hỗ trợ coil write/read, holding register, với **simulation fallback mode** khi không connect được.
2. **`VirtualModbusServer`** (`src/industrial/simulator_plc.py`): Server Modbus giả lập chạy local, dùng cho test integration và development.

## Decision drivers
- **pymodbus** là thư viện Python Modbus phổ biến, đã có trong requirements.txt.
- Virtual simulator cho phép test end-to-end tanpa hardware — quan trọng cho CI/CD và developer onboarding.
- Simulation fallback đảm bảo system không crash khi PLC offline (graceful degradation).

## Trade-offs
- Virtual simulator không phải real PLC — timing, error behavior khác hardware thật.
- Modbus TCP là protocol cũ, không có encryption/auth built-in — cần network-level security (VLAN, firewall).
- Simulation mode phải được flag rõ ràng để tránh confuse operator.

## Consequences
- Test `test_plc_bridge.py` cover cả simulation mode và virtual server.
- Production deployment phải document cách cấu hình PLC thật + network.
- Fallback mode ghi log rõ ràng: `SIMULATED` vs `DISPATCHED`.

---

# docs/adr/003-onnx-canonical-weights.md
# ADR-003: ONNX Runtime + Canonical Colab-trained Weights cho Edge Inference

## Status
Accepted (2026-09-16)

## Context
Cần inference engine cho PCB defect detection:

1. **PyTorch/live YOLOv8**: cần GPU, heavy, không phù hợp edge deployment.
2. **ONNX Runtime CPU**: lightweight, cross-platform, đã optimized cho CPU inference.
3. **TFLite/Edge TPU**: cần hardware-specific, chưa availableTrong project.

## Decision
Chọn **ONNX Runtime CPU** làm inference backend, với model weights là **YOLOv8n được train trên Google Colab T4 và export sang ONNX**.

Model canonical: `models/yolov8n_pcb_defect.onnx` (~12 MB, git-tracked).

### Why ONNX?
- **CPU-optimized**: chạy được trên edge device không GPU (cloud 2-vCPU target).
- **Cross-platform**: identical behavior trên macOS, Linux, Windows.
- **Lightweight**: 12 MB vs PyTorch 전체 (수백 MB).
- **Deterministic**: same input → same output, quan trọng cho industrial 품질 này.

### Why Colab-trained?
- Training trên Colab GPU T4 miễn phí, reproducible.
- Notebook `notebooks/train_pcb_defect_yolo.ipynb` là evidence frozen.
- Export được verify IO shapes: input `[1,3,640,640]` → output `[1,10,8400]`.

## Trade-offs
- ONNX export có thể mất một số dynamic op nếu model complex — nhưng YOLOv8 export ổn định.
- CPU inference chậm hơn GPU — nhưng vẫn đạt target ≤35ms trên hardware phù hợp.
- Không real-time 4K — phải dùng SAHI patching cho high-res (đã implement `HighResPatchInferencer`).

## Consequences
- `PCBDefectDetector` load ONNX session, preprocess, postprocess YOLOv8 NMS.
- Model path resolution chain: arg → `$MODEL_PATH` → `models/yolov8n_pcb_defect.onnx`.
- Fallback GT chỉ khi `ALLOW_GT_FALLBACK=true` (default false — production-safe).
- PERFORMANCE: per-frame `inference_time_ms` đo real-time và ghi vào DB.