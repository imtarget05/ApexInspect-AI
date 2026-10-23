"""
High-performance predictor for defect detection.
Supports PyTorch, ONNX Runtime, and TensorRT backends.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch


class Backend(Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"


@dataclass
class PredictorConfig:
    """Configuration for predictor."""
    model_path: str
    backend: Backend = Backend.PYTORCH
    device: str = "auto"  # auto, cuda, cpu
    img_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 300
    class_names: Optional[List[str]] = None
    half_precision: bool = False
    batch_size: int = 1
    
    # ONNX/TensorRT specific
    providers: List[str] = field(default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    
    # TensorRT specific
    workspace_size: int = 4 << 30  # 4GB
    precision: str = "fp16"  # fp32, fp16, int8


class DefectPredictor:
    """
    Unified predictor for defect detection.
    Automatically selects backend based on model file extension.
    """
    
    def __init__(self, config: PredictorConfig):
        self.config = config
        self.backend = self._detect_backend(config.model_path)
        self.device = self._get_device(config.device)
        self.model = None
        self.session = None
        self.context = None
        self.bindings = None
        self.input_name = None
        self.output_names = None
        
        self._load_model()
        self._warmup()
    
    def _detect_backend(self, model_path: str) -> Backend:
        """Auto-detect backend from file extension."""
        ext = Path(model_path).suffix.lower()
        if ext in [".pt", ".pth"]:
            return Backend.PYTORCH
        elif ext == ".onnx":
            return Backend.ONNX
        elif ext in [".engine", ".plan"]:
            return Backend.TENSORRT
        elif ext == ".xml":
            return Backend.OPENVINO
        return Backend.PYTORCH
    
    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self):
        """Load model based on backend."""
        if self.backend == Backend.PYTORCH:
            self._load_pytorch()
        elif self.backend == Backend.ONNX:
            self._load_onnx()
        elif self.backend == Backend.TENSORRT:
            self._load_tensorrt()
        elif self.backend == Backend.OPENVINO:
            self._load_openvino()
    
    def _load_pytorch(self):
        from apexinspect.models import DefectDetector
        self.model = DefectDetector.load(self.config.model_path)
        self.model.to(self.device)
        self.model.eval()
        if self.config.half_precision and self.device.type == "cuda":
            self.model.half()
    
    def _load_onnx(self):
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("onnxruntime not installed")
        
        self.session = ort.InferenceSession(
            self.config.model_path,
            providers=self.config.providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
    
    def _load_tensorrt(self):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise RuntimeError("tensorrt and pycuda required for TensorRT backend")
        
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        
        with open(self.config.model_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = engine.create_execution_context()
        self.context.set_input_shape("input", (self.config.batch_size, 3, self.config.img_size, self.config.img_size))
        
        # Allocate buffers
        self.bindings = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            size = np.prod(shape)
            self.bindings.append(cuda.mem_alloc(size * np.dtype(dtype).itemsize))
        
        self.input_name = engine.get_tensor_name(0)
    
    def _load_openvino(self):
        try:
            from openvino.runtime import Core
        except ImportError:
            raise RuntimeError("openvino not installed")
        
        core = Core()
        self.model = core.compile_model(self.config.model_path, "GPU" if self.device.type == "cuda" else "CPU")
        self.input_name = self.model.input(0).get_any_name()
        self.output_names = [o.get_any_name() for o in self.model.outputs()]
    
    def _warmup(self, iterations: int = 3):
        """Warmup model with dummy input."""
        dummy = np.random.randn(self.config.batch_size, 3, self.config.img_size, self.config.img_size).astype(np.float32)
        for _ in range(iterations):
            self.predict(dummy)
    
    def predict(
        self,
        images: Union[np.ndarray, List[np.ndarray], torch.Tensor],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[Dict]:
        """
        Run inference on images.
        
        Args:
            images: Input images (B, 3, H, W) or (3, H, W) or list of (H, W, 3)
            conf: Confidence threshold override
            iou: IoU threshold override
            
        Returns:
            List of dicts with boxes, scores, class_ids, class_names
        """
        conf = conf or self.config.conf_threshold
        iou = iou or self.config.iou_threshold
        
        # Preprocess
        if isinstance(images, list):
            images = np.stack(images)
        elif isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        # Ensure batch dimension
        if images.ndim == 3:
            images = images[np.newaxis, ...]
        
        # Normalize if needed
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0
        
        # CHW format expected
        if images.shape[1] != 3:
            images = images.transpose(0, 3, 1, 2)
        
        # Resize if needed
        if images.shape[2:] != (self.config.img_size, self.config.img_size):
            import cv2
            resized = []
            for img in images:
                resized.append(cv2.resize(
                    img.transpose(1, 2, 0),
                    (self.config.img_size, self.config.img_size)
                ).transpose(2, 0, 1))
            images = np.stack(resized)
        
        # Run inference
        if self.backend == Backend.PYTORCH:
            outputs = self._predict_pytorch(images)
        elif self.backend == Backend.ONNX:
            outputs = self._predict_onnx(images)
        elif self.backend == Backend.TENSORRT:
            outputs = self._predict_tensorrt(images)
        elif self.backend == Backend.OPENVINO:
            outputs = self._predict_openvino(images)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        # Post-process
        return self._postprocess(outputs, conf, iou)
    
    def _predict_pytorch(self, images: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            tensor = torch.from_numpy(images).to(self.device)
            if self.config.half_precision:
                tensor = tensor.half()
            return self.model(tensor)
    
    def _predict_onnx(self, images: np.ndarray) -> List[np.ndarray]:
        return self.session.run(self.output_names, {self.input_name: images})
    
    def _predict_tensorrt(self, images: np.ndarray) -> List[np.ndarray]:
        import pycuda.driver as cuda
        
        # Copy input to device
        cuda.memcpy_htod(self.bindings[0], images)
        
        # Execute
        self.context.execute_v2(self.bindings)
        
        # Copy outputs back
        outputs = []
        for i, binding in enumerate(self.bindings[1:], 1):
            shape = self.context.get_tensor_shape(f"output{i-1}")
            dtype = trt.nptype(self.context.get_tensor_dtype(f"output{i-1}"))
            output = np.empty(shape, dtype=dtype)
            cuda.memcpy_dtoh(output, binding)
            outputs.append(output)
        
        return outputs
    
    def _predict_openvino(self, images: np.ndarray) -> List[np.ndarray]:
        result = self.model({self.input_name: images})
        return [result[name] for name in self.output_names]
    
    def _postprocess(self, outputs, conf: float, iou: float) -> List[Dict]:
        """Post-process raw outputs to detections."""
        # This is YOLO-specific postprocessing
        # For generic predictor, return raw outputs
        return [{"raw_output": out} for out in outputs]
    
    def benchmark(
        self,
        input_shape: tuple = (1, 3, 640, 640),
        warmup: int = 10,
        iterations: int = 100,
    ) -> Dict:
        """Benchmark inference speed."""
        dummy = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup):
            self.predict(dummy)
        
        # Benchmark
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.predict(dummy)
            latencies.append(time.perf_counter() - start)
        
        latencies = np.array(latencies) * 1000  # ms
        
        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p90_ms": float(np.percentile(latencies, 90)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "throughput_fps": float(self.config.batch_size / (np.mean(latencies) / 1000)),
        }
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


class BatchInference:
    """Batch inference with async processing."""
    
    def __init__(self, predictor: DefectPredictor, max_batch_size: int = 32):
        self.predictor = predictor
        self.max_batch_size = max_batch_size
        self.buffer = []
    
    def add(self, image: np.ndarray) -> Optional[List[Dict]]:
        """Add image to buffer, process when batch is full."""
        self.buffer.append(image)
        if len(self.buffer) >= self.max_batch_size:
            return self.flush()
        return None
    
    def flush(self) -> List[Dict]:
        """Process current buffer."""
        if not self.buffer:
            return []
        batch = np.stack(self.buffer)
        results = self.predictor.predict(batch)
        self.buffer = []
        return results


class StreamInference:
    """Streaming inference for video/camera input."""
    
    def __init__(
        self,
        predictor: DefectPredictor,
        source: Union[int, str] = 0,
        buffer_size: int = 10,
    ):
        self.predictor = predictor
        self.source = source
        self.buffer_size = buffer_size
        self.cap = None
        self.running = False
    
    def start(self):
        """Start streaming inference."""
        import cv2
        self.cap = cv2.VideoCapture(self.source)
        self.running = True
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Predict
            results = self.predictor.predict(frame_rgb)
            
            # Visualize
            frame = self._draw_detections(frame, results)
            
            cv2.imshow("ApexInspect Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop()
    
    def stop(self):
        """Stop streaming."""
        self.running = False
        if self.cap:
            self.cap.release()
        import cv2
        cv2.destroyAllWindows()
    
    def _draw_detections(self, frame, results):
        import cv2
        for det in results:
            for box, score, cls_id in zip(
                det.get("boxes", []),
                det.get("scores", []),
                det.get("class_ids", [])
            ):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{self.predictor.config.class_names[cls_id] if self.predictor.config.class_names else cls_id}: {score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame


__all__ = [
    "Backend",
    "PredictorConfig",
    "DefectPredictor",
    "BatchInference",
    "StreamInference",
]