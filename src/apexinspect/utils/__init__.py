"""Utility functions for ApexInspect-AI."""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional
import logging
from logging.handlers import RotatingFileHandler
import structlog


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device: str = "auto") -> torch.device:
    """Get torch device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_type: str = "json",
) -> structlog.BoundLogger:
    """Setup structured logging with structlog."""
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )
    
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        logging.getLogger().addHandler(file_handler)
    
    # Configure structlog
    if format_type == "json":
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()


def get_logger(name: str = "apexinspect") -> structlog.BoundLogger:
    """Get logger instance."""
    return structlog.get_logger(name)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


def model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def save_config(config: dict, path: Union[str, Path]):
    """Save config to YAML."""
    import yaml
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(path: Union[str, Path]) -> dict:
    """Load config from YAML."""
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Union[str, Path]):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_gpu_memory() -> Dict[int, Dict[str, float]]:
    """Get GPU memory usage."""
    try:
        import pynvml
        pynvml.nvmlInit()
        gpus = {}
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus[i] = {
                "total_mb": info.total / (1024 ** 2),
                "used_mb": info.used / (1024 ** 2),
                "free_mb": info.free / (1024 ** 2),
            }
        return gpus
    except:
        return {}


def format_time(seconds: float) -> str:
    """Format time in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def format_bytes(bytes_val: int) -> str:
    """Format bytes in human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}PB"


__all__ = [
    "set_seed",
    "get_device",
    "setup_logging",
    "get_logger",
    "count_parameters",
    "model_size_mb",
    "save_config",
    "load_config",
    "ensure_dir",
    "get_gpu_memory",
    "format_time",
    "format_bytes",
]