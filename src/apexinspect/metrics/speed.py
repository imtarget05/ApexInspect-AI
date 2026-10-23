"""
Speed/latency benchmarking for model inference.
"""

import time
import numpy as np
import torch
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import json
from contextlib import contextmanager


@contextmanager
def timer():
    """Context manager for timing."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def benchmark_model(
    model: Callable,
    input_shape: tuple = (1, 3, 640, 640),
    device: str = "cuda",
    warmup: int = 10,
    iterations: int = 100,
    batch_sizes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Benchmark model inference speed.
    
    Args:
        model: Callable that takes input tensor and returns output
        input_shape: Input tensor shape (B, C, H, W)
        device: Device to run on
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dict with latency stats per batch size
    """
    model.eval()
    model.to(device)
    
    batch_sizes = batch_sizes or [1, 2, 4, 8, 16]
    results = {}

    for batch_size in batch_sizes:
        shape = (batch_size,) + input_shape[1:]
        dummy_input = torch.randn(shape, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

        # Synchronize
        if device == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        latencies = []
        with torch.no_grad():
            for _ in range(iterations):
                if device == "cuda":
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = model(dummy_input)
                
                if device == "cuda":
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - start)

        latencies = np.array(latencies) * 1000  # Convert to ms

        results[batch_size] = {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p90_ms": float(np.percentile(latencies, 90)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "throughput_fps": float(batch_size / (np.mean(latencies) / 1000)),
        }

    return results


def profile_inference(
    model: Callable,
    input_shape: tuple = (1, 3, 640, 640),
    device: str = "cuda",
    detailed: bool = True,
) -> Dict[str, Any]:
    """
    Detailed profiling using PyTorch profiler.
    
    Returns:
        Dict with kernel times, memory usage, etc.
    """
    try:
        from torch.profiler import profile, record_function, ProfilerActivity
    except ImportError:
        return {"error": "torch.profiler not available"}

    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape, device=device)

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=detailed,
        profile_memory=detailed,
        with_stack=detailed,
    ) as prof:
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

    # Get key metrics
    key_averages = prof.key_averages(group_by_input_shape=True)
    
    # Top 10 by CPU time
    top_cpu = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:10]
    
    result = {
        "top_kernels_cpu": [
            {
                "name": k.key,
                "cpu_time_ms": k.cpu_time_total / 1000,
                "cuda_time_ms": k.cuda_time_total / 1000 if hasattr(k, 'cuda_time_total') else 0,
                "count": k.count,
            }
            for k in top_cpu
        ],
    }

    if device == "cuda":
        top_cuda = sorted(key_averages, key=lambda x: x.cuda_time_total, reverse=True)[:10]
        result["top_kernels_cuda"] = [
            {
                "name": k.key,
                "cuda_time_ms": k.cuda_time_total / 1000,
                "count": k.count,
            }
            for k in top_cuda
        ]

    if detailed:
        result["full_trace"] = prof.export_chrome_trace("/tmp/trace.json")

    return result


def benchmark_onnx(
    onnx_path: str,
    input_shape: tuple = (1, 3, 640, 640),
    device: str = "cuda",
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, Any]:
    """Benchmark ONNX model with ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime not installed"}

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})

    # Timed runs
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies.append(time.perf_counter() - start)

    latencies = np.array(latencies) * 1000

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_fps": float(input_shape[0] / (np.mean(latencies) / 1000)),
    }


def benchmark_tensorrt(
    engine_path: str,
    input_shape: tuple = (1, 3, 640, 640),
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, Any]:
    """Benchmark TensorRT engine."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        return {"error": "tensorrt or pycuda not installed"}

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    context.set_input_shape("input", input_shape)

    # Allocate buffers
    bindings = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = np.prod(shape)
        bindings.append(cuda.mem_alloc(size * np.dtype(dtype).itemsize))

    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        cuda.memcpy_htod(bindings[0], dummy_input)
        context.execute_v2(bindings)

    # Timed runs
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        cuda.memcpy_htod(bindings[0], dummy_input)
        context.execute_v2(bindings)
        latencies.append(time.perf_counter() - start)

    latencies = np.array(latencies) * 1000

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_fps": float(input_shape[0] / (np.mean(latencies) / 1000)),
    }


def compare_backends(
    model_paths: Dict[str, str],
    input_shape: tuple = (1, 3, 640, 640),
    device: str = "cuda",
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, Any]:
    """Compare inference speed across backends."""
    results = {}

    for name, path in model_paths.items():
        if path.endswith(".pt") or path.endswith(".pth"):
            # PyTorch
            model = torch.load(path, map_location=device)
            model.eval()
            results[name] = benchmark_model(model, input_shape, device, warmup, iterations)
        elif path.endswith(".onnx"):
            results[name] = benchmark_onnx(path, input_shape, device, warmup, iterations)
        elif path.endswith(".engine") or path.endswith(".plan"):
            results[name] = benchmark_tensorrt(path, input_shape, warmup, iterations)
        else:
            results[name] = {"error": f"Unknown format: {path}"}

    # Summary table
    summary = {}
    for name, res in results.items():
        if "error" not in res and 1 in res:
            b1 = res[1]
            summary[name] = {
                "latency_ms": b1["mean_ms"],
                "p99_ms": b1["p99_ms"],
                "throughput_fps": b1["throughput_fps"],
            }

    return {"details": results, "summary": summary}


def save_benchmark_results(results: Dict, output_path: Union[str, Path]):
    """Save benchmark results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


__all__ = [
    "benchmark_model",
    "profile_inference",
    "benchmark_onnx",
    "benchmark_tensorrt",
    "compare_backends",
    "save_benchmark_results",
]