"""
Model export utilities: PyTorch -> ONNX -> TensorRT -> CoreML/TFLite.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from apexinspect.models import DefectDetector


def export_onnx(
    model: Union[DefectDetector, torch.nn.Module],
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    opset: int = 17,
    dynamic_batch: bool = True,
    simplify: bool = True,
    half: bool = False,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
) -> Path:
    """
    Export PyTorch model to ONNX.
    
    Args:
        model: PyTorch model or DefectDetector
        output_path: Output ONNX file path
        input_shape: Input tensor shape
        opset: ONNX opset version
        dynamic_batch: Enable dynamic batch size
        simplify: Simplify ONNX model
        half: Export as FP16
        input_names: Custom input names
        output_names: Custom output names
        
    Returns:
        Path to exported ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract raw model if DefectDetector
    if isinstance(model, DefectDetector):
        pytorch_model = model.model.model
    else:
        pytorch_model = model
    
    pytorch_model.eval()
    
    # Prepare dummy input
    dummy_input = torch.randn(input_shape)
    if half:
        dummy_input = dummy_input.half()
        pytorch_model.half()
    
    # Default names
    input_names = input_names or ["images"]
    output_names = output_names or ["output"]
    
    # Dynamic axes
    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch"},
            "output": {0: "batch"},
        }
    
    # Export
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    
    # Simplify
    if simplify:
        try:
            import onnx
            from onnxsim import simplify
            
            onnx_model = onnx.load(output_path)
            onnx_model, check = simplify(onnx_model)
            if check:
                onnx.save(onnx_model, output_path)
                print(f"ONNX simplified and saved to {output_path}")
            else:
                print("ONNX simplification failed, keeping original")
        except ImportError:
            print("onnxsim not installed, skipping simplification")
    
    # Verify
    verify_onnx(output_path, input_shape)
    
    return output_path


def verify_onnx(onnx_path: Union[str, Path], input_shape: Tuple[int, ...] = (1, 3, 640, 640)) -> bool:
    """Verify ONNX model loads and runs correctly."""
    try:
        import onnx
        import onnxruntime as ort
        
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Test inference
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        dummy = np.random.randn(*input_shape).astype(np.float32)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: dummy})
        
        print(f"ONNX verification passed. Output shapes: {[o.shape for o in outputs]}")
        return True
    except Exception as e:
        print(f"ONNX verification failed: {e}")
        return False


def export_tensorrt(
    onnx_path: Union[str, Path],
    output_path: Union[str, Path],
    precision: str = "fp16",  # fp32, fp16, int8
    max_batch_size: int = 8,
    min_shape: Tuple[int, ...] = (1, 3, 640, 640),
    opt_shape: Tuple[int, ...] = (4, 3, 640, 640),
    max_shape: Tuple[int, ...] = (8, 3, 640, 640),
    workspace_size: int = 4 << 30,  # 4GB
    calibration_cache: Optional[str] = None,
    calibration_data_loader: Optional[callable] = None,
) -> Path:
    """
    Convert ONNX to TensorRT engine.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Output TensorRT engine path
        precision: fp32, fp16, or int8
        max_batch_size: Maximum batch size
        min_shape: Minimum input shape for optimization profile
        opt_shape: Optimal input shape
        max_shape: Maximum input shape
        workspace_size: GPU workspace size in bytes
        calibration_cache: Path to INT8 calibration cache
        calibration_data_loader: Function that yields calibration data for INT8
        
    Returns:
        Path to TensorRT engine
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise RuntimeError("TensorRT not installed. Install from NVIDIA container or pip.")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    
    # Precision
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        if calibration_cache:
            config.set_calibration_profile(
                trt.CalibrationProfile(calibration_cache)
            )
        if calibration_data_loader:
            # Set up calibrator
            from apexinspect.deployment.calibration import EntropyCalibrator
            calibrator = EntropyCalibrator(calibration_data_loader, calibration_cache)
            config.int8_calibrator = calibrator
    
    # Parse ONNX
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX Parser Error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")
    
    # Optimization profile
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    
    profile.set_shape(
        input_name,
        min_shape,
        opt_shape,
        max_shape,
    )
    config.add_optimization_profile(profile)
    
    # Build engine
    print(f"Building TensorRT engine (precision={precision})...")
    engine = builder.build_serialized_network(network, config)
    
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    # Save
    with open(output_path, "wb") as f:
        f.write(engine)
    
    print(f"TensorRT engine saved to {output_path}")
    return output_path


def export_coreml(
    model: Union[DefectDetector, torch.nn.Module],
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    class_labels: Optional[List[str]] = None,
    compute_precision: str = "float16",  # float16, float32
) -> Path:
    """
    Export to CoreML format for iOS/macOS deployment.
    
    Args:
        model: PyTorch model
        output_path: Output .mlmodel path
        input_shape: Input shape
        class_labels: Class labels for detection
        compute_precision: float16 or float32
    """
    try:
        import coremltools as ct
    except ImportError:
        raise RuntimeError("coremltools not installed")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(model, DefectDetector):
        pytorch_model = model.model.model
    else:
        pytorch_model = model
    
    pytorch_model.eval()
    
    # Trace model
    dummy = torch.randn(input_shape)
    traced = torch.jit.trace(pytorch_model, dummy)
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=input_shape)],
        classifier_config=None,
        compute_precision=ct.precision.FLOAT16 if compute_precision == "float16" else ct.precision.FLOAT32,
    )
    
    # Add metadata
    if class_labels:
        mlmodel.short_description = "ApexInspect Industrial Defect Detection"
        mlmodel.version = "1.0"
        mlmodel.author = "ApexInspect-AI"
        mlmodel.license = "Proprietary"
    
    mlmodel.save(output_path)
    print(f"CoreML model saved to {output_path}")
    return output_path


def export_tflite(
    model: Union[DefectDetector, torch.nn.Module],
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    quantize: bool = True,
    representative_dataset: Optional[callable] = None,
) -> Path:
    """
    Export to TensorFlow Lite format for mobile/edge deployment.
    
    Args:
        model: PyTorch model
        output_path: Output .tflite path
        input_shape: Input shape
        quantize: Enable INT8 quantization
        representative_dataset: Generator for quantization calibration
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError("TensorFlow not installed")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(model, DefectDetector):
        pytorch_model = model.model.model
    else:
        pytorch_model = model
    
    pytorch_model.eval()
    
    # Convert PyTorch -> ONNX -> TF -> TFLite
    # This is a multi-step process
    
    # Step 1: PyTorch to ONNX
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    
    export_onnx(pytorch_model, onnx_path, input_shape)
    
    # Step 2: ONNX to TF
    import onnx
    from onnx_tf.backend import prepare
    
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tf_path = Path(tmpdir) / "saved_model"
        tf_rep.export_graph(str(tf_path))
        
        # Step 3: TF to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if representative_dataset:
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        
        with open(output_path, "wb") as f:
            f.write(tflite_model)
    
    # Cleanup
    Path(onnx_path).unlink(missing_ok=True)
    
    print(f"TFLite model saved to {output_path}")
    return output_path


def optimize_onnx(
    onnx_path: Union[str, Path],
    output_path: Union[str, Path],
    optimization_level: int = 99,
) -> Path:
    """Optimize ONNX model with onnx-simplifier and onnxruntime."""
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        raise RuntimeError("onnxsim not installed")
    
    onnx_model = onnx.load(onnx_path)
    onnx_model, check = simplify(onnx_model, perform_optimization=True)
    
    if check:
        onnx.save(onnx_model, output_path)
        print(f"Optimized ONNX saved to {output_path}")
    else:
        print("Optimization failed, copying original")
        import shutil
        shutil.copy(onnx_path, output_path)
    
    return output_path


def fuse_onnx(onnx_path: Union[str, Path], output_path: Union[str, Path]) -> Path:
    """Fuse Conv+BN and other patterns in ONNX."""
    try:
        import onnx
        from onnx import optimizer
    except ImportError:
        raise RuntimeError("onnx optimizer not available")
    
    onnx_model = onnx.load(onnx_path)
    
    # Apply optimization passes
    passes = [
        "fuse_bn_into_conv",
        "fuse_add_bias_into_conv",
        "fuse_consecutive_transposes",
        "eliminate_deadend",
        "eliminate_identity",
        "eliminate_nop_pad",
        "eliminate_nop_transpose",
        "eliminate_unused_initializer",
    ]
    
    optimized = optimizer.optimize(onnx_model, passes)
    onnx.save(optimized, output_path)
    
    return output_path


__all__ = [
    "export_onnx",
    "verify_onnx",
    "export_tensorrt",
    "export_coreml",
    "export_tflite",
    "optimize_onnx",
    "fuse_onnx",
]