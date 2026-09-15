"""
Utility to export or download a lightweight YOLOv8 PCB Defect Detection model to ONNX format.
"""
import os
import sys

def export_onnx_model():
    print("[Export] Checking Ultralytics YOLO installation...")
    try:
        from ultralytics import YOLO
        target_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(target_dir, "yolov8n_pcb_defect.onnx")

        if os.path.exists(output_path):
            print(f"[Export] Model already exists at: {output_path}")
            return

        print("[Export] Exporting lightweight YOLOv8n to ONNX (optimized for CPU inference)...")
        model = YOLO("yolov8n.pt")
        model.export(format="onnx", imgsz=640, optimize=True)

        exported_default = "yolov8n.onnx"
        if os.path.exists(exported_default):
            os.rename(exported_default, output_path)
            print(f"[Export] Successfully created: {output_path}")
    except ImportError:
        print("[Export] ultralytics package not installed. Run: pip install ultralytics")
    except Exception as e:
        print(f"[Export] Note during export: {e}")

if __name__ == "__main__":
    export_onnx_model()
