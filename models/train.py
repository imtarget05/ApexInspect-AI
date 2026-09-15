"""
CLI Training Script for Custom YOLOv8 PCB Defect Detection Model.
Can be executed on Colab, Kaggle, or a local GPU machine.
"""
import os
import sys
import shutil

def train_custom_pcb_model(epochs: int = 50, batch_size: int = 16, imgsz: int = 640):
    print("=" * 60)
    print("🏭 ApexInspect AI — Custom YOLOv8 PCB Training Pipeline")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics is not installed. Please run: pip install ultralytics onnx")
        return

    # 1. Prepare data.yaml if not present
    data_yaml = "data.yaml"
    if not os.path.exists(data_yaml):
        print("📝 Generating default data.yaml for PCB defect dataset...")
        content = """
path: ./dataset_pcb
train: images/train
val: images/val
test: images/val

names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
"""
        with open(data_yaml, "w") as f:
            f.write(content.strip())

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
        name="yolov8n_pcb_custom"
    )

    # 3. Export to ONNX
    best_weights = "runs/apex_inspect/yolov8n_pcb_custom/weights/best.pt"
    if os.path.exists(best_weights):
        print(f"📦 Exporting trained weights {best_weights} to ONNX format...")
        trained_model = YOLO(best_weights)
        onnx_file = trained_model.export(format="onnx", imgsz=imgsz, optimize=True)

        target_dest = os.path.join(os.path.dirname(__file__), "yolov8n_pcb_defect.onnx")
        shutil.copy(onnx_file, target_dest)
        print(f"🎉 Model deployed to target destination: {target_dest}")
    else:
        print("⚠️ best.pt not found, exporting base yolov8n as fallback...")
        model.export(format="onnx", imgsz=imgsz, optimize=True)

if __name__ == "__main__":
    train_custom_pcb_model()
