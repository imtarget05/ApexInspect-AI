"""
FastAPI serving for defect detection model.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import io
import base64
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from apexinspect.inference import DefectPredictor, PredictorConfig, Backend


@dataclass
class ModelConfig:
    """Model configuration for FastAPI app."""
    model_path: str
    backend: str = "pytorch"
    device: str = "auto"
    img_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 300
    class_names: Optional[List[str]] = None
    half_precision: bool = False


class DetectionRequest(BaseModel):
    """Single image detection request."""
    image_base64: str = Field(..., description="Base64 encoded image")
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    return_image: bool = False


class BatchDetectionRequest(BaseModel):
    """Batch detection request."""
    images_base64: List[str] = Field(..., description="List of base64 encoded images")
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None


class DetectionResponse(BaseModel):
    """Detection response."""
    boxes: List[List[float]]
    scores: List[float]
    class_ids: List[int]
    class_names: List[str]
    inference_time_ms: float
    image_shape: List[int]


class BatchDetectionResponse(BaseModel):
    """Batch detection response."""
    detections: List[DetectionResponse]
    total_inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_path: str
    backend: str
    device: str


def create_app(config: ModelConfig) -> FastAPI:
    """Create FastAPI app with model."""
    
    app = FastAPI(
        title="ApexInspect-AI Defect Detection API",
        description="Industrial defect detection using YOLO",
        version="0.1.0",
    )
    
    # Initialize predictor
    predictor_config = PredictorConfig(
        model_path=config.model_path,
        backend=Backend(config.backend),
        device=config.device,
        img_size=config.img_size,
        conf_threshold=config.conf_threshold,
        iou_threshold=config.iou_threshold,
        max_det=config.max_det,
        class_names=config.class_names,
        half_precision=config.half_precision,
    )
    
    predictor = DefectPredictor(predictor_config)
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_path=config.model_path,
            backend=config.backend,
            device=str(predictor.device),
        )
    
    @app.post("/detect", response_model=DetectionResponse)
    async def detect(request: DetectionRequest):
        """Detect defects in single image."""
        try:
            # Decode base64 image
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_np = np.array(image)
            
            # Run inference
            import time
            start = time.perf_counter()
            results = predictor.predict(
                image_np,
                conf=request.conf_threshold,
                iou=request.iou_threshold,
            )
            inference_time = (time.perf_counter() - start) * 1000
            
            # Format response
            if results and len(results) > 0:
                det = results[0]
                return DetectionResponse(
                    boxes=det.get("boxes", []),
                    scores=det.get("scores", []),
                    class_ids=det.get("class_ids", []),
                    class_names=det.get("class_names", []),
                    inference_time_ms=inference_time,
                    image_shape=list(image_np.shape),
                )
            else:
                return DetectionResponse(
                    boxes=[],
                    scores=[],
                    class_ids=[],
                    class_names=[],
                    inference_time_ms=inference_time,
                    image_shape=list(image_np.shape),
                )
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/detect/batch", response_model=BatchDetectionResponse)
    async def detect_batch(request: BatchDetectionRequest):
        """Detect defects in batch of images."""
        try:
            images = []
            shapes = []
            for img_b64 in request.images_base64:
                image_data = base64.b64decode(img_b64)
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                image_np = np.array(image)
                images.append(image_np)
                shapes.append(list(image_np.shape))
            
            import time
            start = time.perf_counter()
            results = predictor.predict(
                np.stack(images),
                conf=request.conf_threshold,
                iou=request.iou_threshold,
            )
            total_time = (time.perf_counter() - start) * 1000
            
            detections = []
            for i, det in enumerate(results):
                detections.append(DetectionResponse(
                    boxes=det.get("boxes", []),
                    scores=det.get("scores", []),
                    class_ids=det.get("class_ids", []),
                    class_names=det.get("class_names", []),
                    inference_time_ms=total_time / len(images),
                    image_shape=shapes[i],
                ))
            
            return BatchDetectionResponse(
                detections=detections,
                total_inference_time_ms=total_time,
            )
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/detect/file")
    async def detect_file(
        file: UploadFile = File(...),
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ):
        """Detect defects from uploaded file."""
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_np = np.array(image)
            
            import time
            start = time.perf_counter()
            results = predictor.predict(
                image_np,
                conf=conf_threshold,
                iou=iou_threshold,
            )
            inference_time = (time.perf_counter() - start) * 1000
            
            return DetectionResponse(
                boxes=results[0].get("boxes", []) if results else [],
                scores=results[0].get("scores", []) if results else [],
                class_ids=results[0].get("class_ids", []) if results else [],
                class_names=results[0].get("class_names", []) if results else [],
                inference_time_ms=inference_time,
                image_shape=list(image_np.shape),
            )
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/model/info")
    async def model_info():
        """Get model information."""
        return {
            "model_path": config.model_path,
            "backend": config.backend,
            "img_size": config.img_size,
            "conf_threshold": config.conf_threshold,
            "iou_threshold": config.iou_threshold,
            "max_det": config.max_det,
            "class_names": config.class_names,
        }
    
    return app


def serve_model(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    backend: str = "pytorch",
    device: str = "auto",
    img_size: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_names: Optional[List[str]] = None,
    workers: int = 1,
):
    """Serve model with FastAPI."""
    
    config = ModelConfig(
        model_path=model_path,
        backend=backend,
        device=device,
        img_size=img_size,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        class_names=class_names,
    )
    
    app = create_app(config)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


# Example usage:
# python -m apexinspect.deployment.fastapi \
#     --model-path models/onnx/model.onnx \
#     --backend onnx \
#     --port 8000


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Serve ApexInspect model with FastAPI")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx", "tensorrt"])
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, dest="conf_threshold")
    parser.add_argument("--iou", type=float, default=0.45, dest="iou_threshold")
    parser.add_argument("--workers", type=int, default=1)
    
    args = parser.parse_args()
    
    serve_model(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        backend=args.backend,
        device=args.device,
        img_size=args.img_size,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        workers=args.workers,
    )


__all__ = [
    "ModelConfig",
    "DetectionRequest",
    "BatchDetectionRequest",
    "DetectionResponse",
    "BatchDetectionResponse",
    "HealthResponse",
    "create_app",
    "serve_model",
]