"""
Reproducible training pipeline for the YOLOv8 PCB defect model.

Replaces what used to live only as executed cells in the (now frozen)
`notebooks/train_pcb_defect_yolo.ipynb`: build the board-grouped dataset,
train YOLOv8n, and export the canonical ONNX weights.

Usage (GPU/Colab):
    python models/train.py --src /path/to/PCB_DATASET --epochs 50
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

DEFAULT_DATA_YAML = os.path.join(HERE, "training_data.yaml")
DEFAULT_ONNX_DEST = os.path.join(HERE, "yolov8n_pcb_defect.onnx")
DEFAULT_PT_DEST = os.path.join(HERE, "yolov8n_pcb_defect.pt")


def resolve_device():
    """Pick CUDA when available, otherwise CPU.

    Hardcoding `device=0` (as this script used to) makes training crash on any
    CPU-only host, which is how `review-plan-04` flagged it.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def ensure_data_yaml(data_yaml: str, src: str = None,
                     val_ratio: float = 0.2, prod_ratio: float = 0.2) -> str:
    """Use the tracked config when fresh; otherwise build one from `src`.

    `models/training_data.yaml` points at a dataset built by an older,
    image-split run, so when `--src` is given the dataset is rebuilt with
    `scripts/prepare_pcb_dataset.py` (board-grouped, no train/val leakage)
    and the generated `data.yaml` becomes the training config. Without
    `--src`, the previous behaviour is kept: train against the checked-in
    config.
    """
    if src:
        from scripts.prepare_pcb_dataset import build_yolo_dataset
        out = os.path.join(HERE, "dataset_pcb")
        summary = build_yolo_dataset(src, out, val_ratio=val_ratio,
                                     prod_ratio=prod_ratio, seed=42)
        print(f"[train] rebuilt board-grouped dataset: {summary['counts']}")
        return os.path.join(out, "data.yaml")
    if os.path.exists(data_yaml):
        return data_yaml
    raise SystemExit(f"[train] {data_yaml} not found - pass --src <PCB_DATASET> to build it.")


def train_custom_pcb_model(epochs: int = 50, batch_size: int = 16, imgsz: int = 640,
                           data_yaml: str = DEFAULT_DATA_YAML, src: str = None):
    print("=" * 60)
    print("🏭 ApexInspect AI — Custom YOLOv8 PCB Training Pipeline")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics is not installed. Please run: pip install ultralytics onnx")
        return

    # 1. Resolve the data config (checked-in file, or freshly built from --src).
    data_yaml = ensure_data_yaml(data_yaml, src=src)
    device = resolve_device()
    print(f"[train] config: {data_yaml}")
    print(f"[train] device: {device}")

    # 2. Train YOLOv8n
    print(f"🚀 Initializing training: {epochs} epochs, batch {batch_size}, img size {imgsz}...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=10,
        project="runs/apex_inspect",
        name="yolov8n_pcb_custom",
        device=device,
        optimizer="AdamW",
        lr0=0.001,
        mosaic=1.0,
    )

    # 3. Export to ONNX + keep the .pt next to it for future fine-tuning.
    best_weights = "runs/apex_inspect/yolov8n_pcb_custom/weights/best.pt"
    if os.path.exists(best_weights):
        print(f"📦 Exporting trained weights {best_weights} to ONNX format...")
        trained_model = YOLO(best_weights)
        onnx_file = trained_model.export(format="onnx", imgsz=imgsz)
        shutil.copy(onnx_file, DEFAULT_ONNX_DEST)
        shutil.copy(best_weights, DEFAULT_PT_DEST)
        print(f"🎉 Model deployed: {DEFAULT_ONNX_DEST} + {DEFAULT_PT_DEST}")
    else:
        print("⚠️ best.pt not found, exporting base yolov8n as fallback...")
        model.export(format="onnx", imgsz=imgsz)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train YOLOv8 PCB defect model (see module docstring)")
    ap.add_argument("--src", default=None, help="PCB_DATASET root; rebuilds the board-grouped dataset first")
    ap.add_argument("--data-yaml", default=DEFAULT_DATA_YAML)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    train_custom_pcb_model(epochs=args.epochs, batch_size=args.batch_size,
                           imgsz=args.imgsz, data_yaml=args.data_yaml, src=args.src)
