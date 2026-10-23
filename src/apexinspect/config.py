"""
Configuration management using Hydra + OmegaConf.
Supports YAML configs with environment variable overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import os


@dataclass
class ModelConfig:
    name: str = "yolov8n"
    num_classes: int = 10
    input_size: int = 640
    pretrained: bool = True
    freeze_backbone: bool = False
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 300


@dataclass
class DataConfig:
    train_path: str = "data/yolo_format/train"
    val_path: str = "data/yolo_format/val"
    test_path: str = "data/yolo_format/test"
    class_names: List[str] = field(default_factory=lambda: [
        "scratch", "dent", "crack", "corrosion", "misalignment",
        "contamination", "burr", "deformation", "discoloration", "foreign_object"
    ])
    batch_size: int = 16
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    augment: bool = True
    mosaic: float = 1.0
    mixup: float = 0.1
    copy_paste: float = 0.3


@dataclass
class AugmentationConfig:
    # Albumentations config
    blur_prob: float = 0.1
    median_blur_prob: float = 0.1
    to_gray_prob: float = 0.1
    clahe_prob: float = 0.1
    noise_prob: float = 0.2
    brightness_contrast_prob: float = 0.3
    hue_saturation_prob: float = 0.2
    flip_prob: float = 0.5
    rotate_limit: int = 30
    scale_limit: float = 0.5
    translate_limit: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 100
    patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    momentum: float = 0.937
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    box_loss_gain: float = 7.5
    cls_loss_gain: float = 0.5
    dfl_loss_gain: float = 1.5
    label_smoothing: float = 0.0
    nbs: int = 64  # nominal batch size
    overlap_mask: bool = True
    mask_ratio: int = 4
    dropout: float = 0.0
    val_interval: int = 1
    save_period: int = 10


@dataclass
class OptimizerConfig:
    name: str = "SGD"
    lr: float = 1e-3
    momentum: float = 0.937
    weight_decay: float = 5e-4
    nesterov: bool = True


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    lr_final: float = 1e-5
    warmup_epochs: int = 3


@dataclass
class ExportConfig:
    format: str = "onnx"  # onnx, torchscript, tensorrt, coreml, tflite
    opset_version: int = 17
    dynamic_batch: bool = True
    simplify: bool = True
    half_precision: bool = True
    workspace_size: int = 4 << 30  # 4GB


@dataclass
class TensorRTConfig:
    precision: str = "fp16"  # fp32, fp16, int8
    max_batch_size: int = 8
    min_shape: List[int] = field(default_factory=lambda: [1, 3, 640, 640])
    opt_shape: List[int] = field(default_factory=lambda: [4, 3, 640, 640])
    max_shape: List[int] = field(default_factory=lambda: [8, 3, 640, 640])
    calibration_cache: str = "models/calibration.cache"


@dataclass
class DeploymentConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    max_batch_size: int = 8
    max_queue_delay: int = 10  # ms
    enable_triton: bool = True
    triton_http_port: int = 8000
    triton_grpc_port: int = 8001
    triton_metrics_port: int = 8002


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/apexinspect.log"
    mlflow_uri: str = "http://localhost:5000"
    wandb_project: str = "apexinspect"
    wandb_entity: Optional[str] = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    tensorrt: TensorRTConfig = field(default_factory=TensorRTConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    device: str = "auto"  # auto, cuda, cpu
    output_dir: str = "outputs"
    experiment_name: str = "apexinspect-industrial"


def load_config(config_path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    """Load configuration from YAML with optional overrides."""
    if config_path and Path(config_path).exists():
        cfg = OmegaConf.load(config_path)
        # Merge with defaults
        default_cfg = OmegaConf.structured(Config)
        cfg = OmegaConf.merge(default_cfg, cfg)
    else:
        cfg = OmegaConf.structured(Config)

    # Apply overrides
    if overrides:
        for override in overrides:
            cfg.merge_with_dotlist([override])

    # Resolve environment variables
    cfg = _resolve_env_vars(cfg)

    return OmegaConf.to_object(cfg)


def _resolve_env_vars(cfg: DictConfig) -> DictConfig:
    """Resolve ${env:VAR} placeholders in config."""
    def resolve_value(value):
        if isinstance(value, str) and value.startswith("${env:"):
            var_name = value[6:-1]  # Remove ${env: and }
            return os.getenv(var_name, value)
        return value

    return OmegaConf.to_container(cfg, resolve=True)


def save_config(cfg: Config, path: str):
    """Save config to YAML."""
    OmegaConf.save(OmegaConf.structured(cfg), path)


# Hydra config registration
def register_configs():
    """Register configs with Hydra."""
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)