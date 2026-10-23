"""
ApexInspect-AI: Industrial Computer Vision Inspection System

A production-ready pipeline for defect detection in manufacturing:
- YOLOv8/v11 fine-tuning
- ONNX/TensorRT optimization
- Triton/FastAPI deployment
- Latency/accuracy benchmarking
"""

__version__ = "0.1.0"
__author__ = "imtarget05"
__email__ = "imtarget05@gmail.com"

from apexinspect.config import load_config
from apexinspect.data import DefectDataset, create_dataloaders
from apexinspect.models import DefectDetector
from apexinspect.metrics import compute_map, compute_f1_per_class

__all__ = [
    "load_config",
    "DefectDataset",
    "create_dataloaders",
    "DefectDetector",
    "compute_map",
    "compute_f1_per_class",
]