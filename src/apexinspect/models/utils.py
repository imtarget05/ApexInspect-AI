"""Model utilities: loading, saving, fusing."""

from pathlib import Path
from typing import Dict, Optional, Union
import torch
from ultralytics import YOLO


def load_weights(
    model: torch.nn.Module,
    path: Union[str, Path],
    strict: bool = True,
    map_location: str = "cpu",
) -> torch.nn.Module:
    """Load weights from checkpoint."""
    checkpoint = torch.load(path, map_location=map_location)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle DDP/DP wrapped models
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Filter by shape
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state[k] = v
            elif not strict:
                print(f"Shape mismatch for {k}: {v.shape} vs {model_state[k].shape}")
        elif not strict:
            print(f"Unexpected key: {k}")

    missing, unexpected = model.load_state_dict(filtered_state, strict=strict)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    return model


def save_weights(
    model: torch.nn.Module,
    path: Union[str, Path],
    epoch: int = 0,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    metrics: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
):
    """Save model checkpoint with metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
    }

    if optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler:
        checkpoint["scheduler"] = scheduler.state_dict()
    if metrics:
        checkpoint["metrics"] = metrics
    if metadata:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, path)


def fuse_model(model: torch.nn.Module) -> torch.nn.Module:
    """Fuse Conv+BN layers for faster inference."""
    from ultralytics.nn.modules import Conv
    
    for m in model.modules():
        if isinstance(m, Conv) and hasattr(m, "bn"):
            m.conv = fuse_conv_bn(m.conv, m.bn)
            m.bn = torch.nn.Identity()
            m.forward = m.forward_fuse
    return model


def fuse_conv_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> torch.nn.Conv2d:
    """Fuse Conv2d and BatchNorm2d."""
    fused_conv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        True,  # bias
    ).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))

    fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.shape))
    fused_conv.bias.copy_(bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps)))

    return fused_conv


def get_model_info(model: torch.nn.Module, input_size: tuple = (1, 3, 640, 640)) -> Dict:
    """Get model info: params, FLOPs, size."""
    from torchinfo import summary

    model_info = summary(model, input_size=input_size, verbose=0)
    return {
        "total_params": model_info.total_params,
        "trainable_params": model_info.trainable_params,
        "total_mult_adds": model_info.total_mult_adds,
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2),
    }


def export_onnx(
    model: torch.nn.Module,
    output_path: Union[str, Path],
    input_size: tuple = (1, 3, 640, 640),
    opset: int = 17,
    dynamic_batch: bool = True,
    simplify: bool = True,
) -> Path:
    """Export PyTorch model to ONNX."""
    import onnx
    from onnxsim import simplify as onnx_simplify

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(input_size)

    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "output": {0: "batch"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    if simplify:
        model_onnx = onnx.load(output_path)
        model_onnx, check = onnx_simplify(model_onnx)
        if check:
            onnx.save(model_onnx, output_path)
        else:
            print("ONNX simplification failed, keeping original")

    return output_path


__all__ = [
    "load_weights",
    "save_weights",
    "fuse_model",
    "fuse_conv_bn",
    "get_model_info",
    "export_onnx",
]