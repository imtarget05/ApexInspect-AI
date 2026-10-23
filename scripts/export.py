#!/usr/bin/env python3
"""
Export model to ONNX, TensorRT, CoreML, TFLite formats.
"""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apexinspect.models import DefectDetector, create_model
from apexinspect.deployment import export_onnx, export_tensorrt, export_coreml, export_tflite, verify_onnx


def main():
    parser = argparse.ArgumentParser(description="Export ApexInspect model to various formats")
    parser.add_argument("--model", type=Path, required=True, help="Input model path (.pt)")
    parser.add_argument("--output-dir", type=Path, default=Path("models/exported"), help="Output directory")
    parser.add_argument("--format", nargs="+", choices=["onnx", "tensorrt", "coreml", "tflite", "all"], default=["onnx"], help="Export formats")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--workspace-size", type=int, default=4, help="Workspace size in GB")
    parser.add_argument("--dynamic-batch", action="store_true", default=True)
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--half", action="store_true", help="Export FP16 ONNX")
    parser.add_argument("--verify", action="store_true", help="Verify exported models")
    parser.add_argument("--class-names", type=Path, help="JSON file with class names")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}")
    model = DefectDetector.load(args.model)
    
    # Get class names
    class_names = None
    if args.class_names:
        import json
        with open(args.class_names) as f:
            class_names = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(args.model).stem
    input_shape = (args.batch_size, 3, args.img_size, args.img_size)
    
    results = {}
    
    # Export ONNX
    if "onnx" in args.format or "all" in args.format:
        print("\n=== Exporting ONNX ===")
        onnx_path = Path(args.output_dir) / f"{base_name}.onnx"
        
        export_onnx(
            model,
            onnx_path,
            input_shape=(args.batch_size, 3, args.img_size, args.img_size),
            opset=args.opset,
            dynamic_batch=args.dynamic_batch,
            simplify=args.simplify,
            half=args.half,
        )
        
        if args.verify:
            verify_onnx(onnx_path, (1, 3, args.img_size, args.img_size))
        
        results["onnx"] = str(onnx_path)
    
    # Export TensorRT
    if "tensorrt" in args.format or "all" in args.format:
        print("\n=== Exporting TensorRT ===")
        onnx_path = Path(args.output_dir) / f"{base_name}.onnx"
        
        # First export ONNX if not already done
        if not onnx_path.exists():
            export_onnx(
                model,
                onnx_path,
                input_shape=(args.batch_size, 3, args.img_size, args.img_size),
                opset=args.opset,
                dynamic_batch=True,
                simplify=True,
                half=args.precision == "fp16",
            )
        
        trt_path = Path(args.output_dir) / f"{base_name}.engine"
        
        export_tensorrt(
            onnx_path=onnx_path,
            output_path=trt_path,
            precision=args.precision,
            max_batch_size=args.max_batch_size,
            min_shape=(1, 3, args.img_size, args.img_size),
            opt_shape=(args.max_batch_size // 2, 3, args.img_size, args.img_size),
            max_shape=(args.max_batch_size, 3, args.img_size, args.img_size),
            workspace_size=args.workspace_size << 30,
        )
        
        if args.verify:
            # Verification would require TensorRT benchmark
            pass
        
        results["tensorrt"] = str(trt_path)
    
    # Export CoreML
    if "coreml" in args.format or "all" in args.format:
        print("\n=== Exporting CoreML ===")
        try:
            coreml_path = Path(args.output_dir) / f"{base_name}.mlmodel"
            
            export_coreml(
                model,
                coreml_path,
                input_shape=(1, 3, args.img_size, args.img_size),
                class_names=class_names,
                compute_precision="float16",
            )
            
            results["coreml"] = str(coreml_path)
        except Exception as e:
            print(f"CoreML export failed: {e}")
    
    # Export TFLite
    if "tflite" in args.format or "all" in args.format:
        print("\n=== Exporting TFLite ===")
        try:
            tflite_path = Path(args.output_dir) / f"{base_name}.tflite"
            
            export_tflite(
                model,
                tflite_path,
                input_shape=(1, 3, args.img_size, args.img_size),
                quantize=True,
            )
            
            results["tflite"] = str(tflite_path)
        except Exception as e:
            print(f"TFLite export failed: {e}")
    
    print("\n=== Export Summary ===")
    for fmt, path in results.items():
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"  {fmt.upper()}: {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()