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

## Trade-offs
- ONNX export có thể mất một số dynamic op nếu model complex — nhưng YOLOv8 export ổn định.
- CPU inference chậm hơn GPU — nhưng vẫn đạt target ≤35ms trên hardware phù hợp.
- Không real-time 4K — phải dùng SAHI patching cho high-res (đã implement `HighResPatchInferencer`).

## Consequences
- `PCBDefectDetector` load ONNX session, preprocess, postprocess YOLOv8 NMS.
- Model path resolution chain: arg → `$MODEL_PATH` → `models/yolov8n_pcb_defect.onnx`.
- Fallback GT chỉ khi `ALLOW_GT_FALLBACK=true` (default false — production-safe).
- PERFORMANCE: per-frame `inference_time_ms` đo real-time và ghi vào DB.