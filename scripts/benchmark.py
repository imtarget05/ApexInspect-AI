#!/usr/bin/env python3
"""
Benchmark script for model inference speed and accuracy.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apexinspect.inference import DefectPredictor, PredictorConfig, Backend
from apexinspect.metrics import benchmark_model, profile_inference, compare_backends
from apexinspect.data import create_dataloaders
from apexinspect.metrics import compute_map, compute_f1_per_class


def main():
    parser = argparse.ArgumentParser(description="Benchmark ApexInspect model")
    parser.add_argument("--model", type=Path, required=True, help="Model path (.pt, .onnx, .engine)")
    parser.add_argument("--data", type=Path, help="Data directory for accuracy evaluation")
    parser.add_argument("--backend", type=str, choices=["pytorch", "onnx", "tensorrt"], help="Backend override")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--profile", action="store_true", help="Run detailed profiling")
    parser.add_argument("--compare", action="store_true", help="Compare multiple backends")
    
    args = parser.parse_args()
    
    # Detect backend from file extension
    if args.backend is None:
        ext = args.model.suffix.lower()
        if ext in [".pt", ".pth"]:
            args.backend = "pytorch"
        elif ext == ".onnx":
            args.backend = "onnx"
        elif ext in [".engine", ".plan"]:
            args.backend = "tensorrt"
        else:
            raise ValueError(f"Unknown model format: {ext}")
    
    print(f"Benchmarking {args.model} with {args.backend} backend")
    
    # Setup predictor
    config = PredictorConfig(
        model_path=str(args.model),
        backend=Backend(args.backend),
        device=args.device,
        img_size=args.img_size,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    
    predictor = DefectPredictor(config)
    
    results = {}
    
    # Speed benchmark
    print("\n=== Speed Benchmark ===")
    speed_results = predictor.benchmark(
        input_shape=(1, 3, config.img_size, config.img_size),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    results["speed"] = speed_results
    
    for bs, metrics in speed_results.items():
        print(f"Batch {bs}: {metrics['mean_ms']:.2f}ms (p99: {metrics['p99_ms']:.2f}ms) | {metrics['throughput_fps']:.1f} FPS")
    
    # Detailed profiling
    if args.profile:
        print("\n=== Detailed Profiling ===")
        profile_results = profile_inference(
            predictor.model if hasattr(predictor, 'model') else None,
            input_shape=(1, 3, config.img_size, config.img_size),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        results["profile"] = profile_results
        if "top_kernels_cuda" in profile_results:
            print("\nTop CUDA Kernels:")
            for k in profile_results["top_kernels_cuda"][:5]:
                print(f"  {k['name']}: {k['cuda_time_ms']:.2f}ms")
    
    # Accuracy evaluation
    if args.data:
        print("\n=== Accuracy Evaluation ===")
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=args.data,
            batch_size=1,
            num_workers=4,
            img_size=config.img_size,
        )
        
        # Run evaluation
        all_preds = []
        all_gts = []
        
        # This would need a full evaluation loop
        print("Accuracy evaluation requires full validation loop")
        results["accuracy"] = {"note": "Requires full validation implementation"}
    
    # Compare backends
    if args.compare:
        print("\n=== Backend Comparison ===")
        model_paths = {
            "PyTorch": str(args.model.with_suffix(".pt")),
            "ONNX": str(args.model.with_suffix(".onnx")),
            "TensorRT": str(args.model.with_suffix(".engine")),
        }
        # Filter existing
        model_paths = {k: v for k, v in model_paths.items() if Path(v).exists()}
        
        if len(model_paths) > 1:
            compare_results = compare_backends(model_paths)
            results["comparison"] = compare_results
            
            print("\n=== Comparison Summary ===")
            for name, metrics in compare_results["summary"].items():
                print(f"  {name}: {metrics['latency_ms']:.2f}ms (p99: {metrics['p99_ms']:.2f}ms) | {metrics['throughput_fps']:.1f} FPS")
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()