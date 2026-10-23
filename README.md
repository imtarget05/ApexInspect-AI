# ApexInspect-AI

> **Industrial Computer Vision Inspection System** — Fine-tuned YOLO detection, ONNX deployment, latency/accuracy benchmarking, and production automation pipeline.

## Overview

ApexInspect-AI is a complete computer vision pipeline for industrial quality inspection. It covers the full lifecycle from dataset preparation → YOLO fine-tuning → ONNX export → TensorRT optimization → edge deployment with latency/accuracy SLA monitoring.

## Features

- **Detection**: YOLOv8/v11 fine-tuning for defect detection (scratches, dents, misalignment, contamination)
- **Data Pipeline**: COCO/YOLO format conversion, augmentation (Albumentations), active learning loop
- **Training**: Distributed training (DDP), mixed precision, hyperparameter optimization (Optuna)
- **Export**: ONNX → TensorRT (FP16/INT8), CoreML, TFLite for edge devices
- **Deployment**: Triton Inference Server, FastAPI service, gRPC, batch/streaming modes
- **Benchmarking**: Latency (p50/p95/p99), throughput, mAP@0.5:0.95, F1 per class
- **Monitoring**: Prometheus/Grafana dashboards, drift detection, auto-retrain triggers

## Quick Start

```bash
# 1. Environment
cd ApexInspect-AI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Prepare data (COCO → YOLO)
python scripts/prepare_data.py --input data/raw --output data/yolo_format

# 3. Train
python src/training/train.py --config configs/yolov8-industrial.yaml

# 4. Export ONNX
python src/deployment/export_onnx.py --weights models/checkpoints/best.pt --output models/onnx/model.onnx

# 5. Benchmark
python src/inference/benchmark.py --model models/onnx/model.onnx --data data/yolo_format/val

# 6. Serve
docker compose up -d triton
```

## Project Structure

```
ApexInspect-AI/
├── configs/              # YAML configs (model, training, deployment)
├── data/
│   ├── raw/              # Original images + annotations
│   ├── processed/        # Cleaned/normalized
│   └── yolo_format/      # train/val/test splits in YOLO format
├── docker/
│   ├── Dockerfile.train  # Training container
│   ├── Dockerfile.infer  # Inference container (TensorRT)
│   └── docker-compose.yml
├── docs/                 # Architecture, API, runbooks
├── models/
│   ├── checkpoints/      # .pt files
│   ├── onnx/             # Exported ONNX
│   └── exported/         # TensorRT engines, CoreML, TFLite
├── notebooks/            # EDA, error analysis, hyperparameter studies
├── scripts/              # One-off utilities (convert, split, visualize)
├── src/
│   ├── apexinspect/      # Core domain (dataset, metrics, IoU)
│   ├── training/         # Train loop, callbacks, DDP utils
│   ├── inference/        # Predictors, benchmarking, TensorRT wrapper
│   ├── deployment/       # ONNX export, Triton config, FastAPI server
│   └── utils/            # Logging, seeding, device management
└── tests/                # Unit + integration tests
```

## Configuration

Key configs in `configs/`:

| File | Purpose |
|------|---------|
| `yolov8-industrial.yaml` | Model arch, loss, optimizer, scheduler |
| `dataset.yaml` | Paths, classes, augmentation pipeline |
| `deployment/triton_config.pbtxt` | Model repository, batching, instance groups |
| `benchmark.yaml` | Batch sizes, warmup runs, precision modes |

## Requirements

- Python 3.10+
- PyTorch 2.3+ (CUDA 12.1+)
- Ultralytics YOLOv8/11
- ONNX Runtime / TensorRT 10+
- Triton Inference Server (optional)
- See `requirements.txt` and `requirements-dev.txt`

## Industrial Automation Integration

- **PLC Interface**: OPC UA client for trigger/capture synchronization
- **MES Export**: JSON/CSV reports with timestamps, defect maps, pass/fail
- **Alerting**: Webhook/Slack/Email on defect rate spike
- **Traceability**: Barcode/QR scan → inspection result linkage

## Metrics & SLA

| Metric | Target | Measurement |
|--------|--------|-------------|
| mAP@0.5 | > 0.95 | COCO eval on holdout test set |
| mAP@0.5:0.95 | > 0.75 | COCO eval |
| Latency p99 | < 15ms | TensorRT FP16, batch=1, Jetson Orin / T4 |
| Throughput | > 100 FPS | Batch=8, TensorRT FP16 |
| Drift detection | Daily | KS-test on prediction confidence distribution |

## License

Proprietary — Internal use only.