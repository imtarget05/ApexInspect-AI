"""
Defect detection model wrapper for YOLOv8/v11.
Supports training, inference, export, and TensorRT optimization.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import intersect_dicts


class DefectDetector(nn.Module):
    """
    Wrapper around YOLO model for industrial defect detection.
    
    Features:
    - Training with custom loss weights
    - Inference with NMS
    - ONNX/TensorRT export
    - Model fusion for faster inference
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        num_classes: int = 10,
        img_size: int = 640,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

        # Load YOLO model
        self.model = YOLO(model_name)
        
        # Modify for custom number of classes if needed
        if pretrained and num_classes != self.model.model.nc:
            self._adapt_num_classes(num_classes)

        if freeze_backbone:
            self._freeze_backbone()

        # Set inference parameters
        self.model.overrides.update({
            "conf": confidence_threshold,
            "iou": iou_threshold,
            "max_det": max_det,
            "imgsz": img_size,
        })

    def _adapt_num_classes(self, num_classes: int):
        """Adapt model head for different number of classes."""
        model = self.model.model
        if isinstance(model, DetectionModel):
            # Get the detection head
            for module in model.modules():
                if hasattr(module, "nc") and module.nc != num_classes:
                    module.nc = num_classes
                    # Reinitialize the last layer
                    if hasattr(module, "cv2"):
                        for cv2 in module.cv2:
                            cv2[-1] = nn.Conv2d(
                                cv2[-1].in_channels,
                                num_classes,
                                cv2[-1].kernel_size,
                                cv2[-1].stride,
                                cv2[-1].padding,
                            )
                    if hasattr(module, "cv3"):
                        for cv3 in module.cv3:
                            cv3[-1] = nn.Conv2d(
                                cv3[-1].in_channels,
                                num_classes * 4,  # 4 for bbox
                                cv3[-1].kernel_size,
                                cv3[-1].stride,
                                cv3[-1].padding,
                            )

    def _freeze_backbone(self):
        """Freeze backbone layers for transfer learning."""
        for name, param in self.model.model.named_parameters():
            if "head" not in name and "cv2" not in name and "cv3" not in name:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def train_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Single training step."""
        return self.model.train_step(batch)

    def val_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Single validation step."""
        return self.model.val_step(batch)

    def predict(
        self,
        source: Union[str, Path, torch.Tensor, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Run inference on images.
        
        Returns:
            List of dicts with: boxes (xyxy), scores, class_ids, class_names
        """
        conf = conf or self.confidence_threshold
        iou = iou or self.iou_threshold

        results = self.model.predict(
            source,
            conf=conf,
            iou=iou,
            max_det=self.max_det,
            imgsz=self.img_size,
            verbose=False,
            **kwargs
        )

        predictions = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else np.empty((0, 4))
            scores = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.empty((0,))
            class_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else np.empty((0,), dtype=int)
            
            class_names = [self.model.names.get(c, str(c)) for c in class_ids]

            predictions.append({
                "boxes": boxes,
                "scores": scores,
                "class_ids": class_ids,
                "class_names": class_names,
            })

        return predictions

    def export_onnx(
        self,
        output_path: Union[str, Path],
        opset: int = 17,
        dynamic_batch: bool = True,
        simplify: bool = True,
        half: bool = False,
    ) -> Path:
        """Export model to ONNX format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.export(
            format="onnx",
            opset=opset,
            dynamic=dynamic_batch,
            simplify=simplify,
            half=half,
            imgsz=self.img_size,
        )

        # Move to desired location
        exported_path = Path(f"{self.model_name}.onnx")
        if exported_path.exists():
            exported_path.rename(output_path)

        return output_path

    def export_tensorrt(
        self,
        onnx_path: Union[str, Path],
        output_path: Union[str, Path],
        precision: str = "fp16",
        max_batch_size: int = 8,
        workspace_size: int = 4 << 30,
    ) -> Path:
        """Export ONNX to TensorRT engine."""
        try:
            import tensorrt as trt
        except ImportError:
            raise RuntimeError("TensorRT not installed. Install with: pip install tensorrt")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

        # Set precision
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            # Need calibration for INT8

        # Parse ONNX
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as f:
            parser.parse(f.read())

        # Build engine
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        profile.set_shape(
            input_name,
            (1, 3, self.img_size, self.img_size),
            (max_batch_size // 2, 3, self.img_size, self.img_size),
            (max_batch_size, 3, self.img_size, self.img_size),
        )
        config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, config)
        with open(output_path, "wb") as f:
            f.write(engine)

        return output_path

    def fuse(self) -> "DefectDetector":
        """Fuse Conv+BN layers for faster inference."""
        self.model.fuse()
        return self

    def to(self, device: Union[str, torch.device]) -> "DefectDetector":
        """Move model to device."""
        self.model.to(device)
        return super().to(device)

    def save(self, path: Union[str, Path]):
        """Save model checkpoint."""
        self.model.save(str(path))

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "DefectDetector":
        """Load model from checkpoint."""
        model = YOLO(str(path))
        detector = cls.__new__(cls)
        detector.model = model
        detector.model_name = Path(path).stem
        detector.num_classes = model.model.nc
        detector.img_size = model.overrides.get("imgsz", 640)
        detector.confidence_threshold = model.overrides.get("conf", 0.25)
        detector.iou_threshold = model.overrides.get("iou", 0.45)
        detector.max_det = model.overrides.get("max_det", 300)
        return detector


def create_model(
    model_name: str = "yolov8n",
    num_classes: int = 10,
    img_size: int = 640,
    pretrained: bool = True,
    **kwargs
) -> DefectDetector:
    """Factory function to create defect detector."""
    return DefectDetector(
        model_name=model_name,
        num_classes=num_classes,
        img_size=img_size,
        pretrained=pretrained,
        **kwargs
    )


def load_weights(model: DefectDetector, path: Union[str, Path], strict: bool = True):
    """Load weights from checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Handle DDP wrapped models
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Filter by shape
    model_state = model.model.model.state_dict()
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered_state[k] = v
        elif not strict:
            print(f"Skipping {k}: shape mismatch {v.shape} vs {model_state.get(k, 'missing')}")

    model.model.model.load_state_dict(filtered_state, strict=strict)
    return model


def save_weights(model: DefectDetector, path: Union[str, Path], epoch: int = 0, metrics: Dict = None):
    """Save model checkpoint with metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model": model.model.model.state_dict(),
        "model_name": model.model_name,
        "num_classes": model.num_classes,
        "img_size": model.img_size,
        "metrics": metrics or {},
    }
    torch.save(checkpoint, path)


def fuse_model(model: DefectDetector) -> DefectDetector:
    """Fuse Conv+BN layers in model."""
    model.fuse()
    return model