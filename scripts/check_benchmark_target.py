#!/usr/bin/env python3
"""
Automated benchmark checker: chạy IPC nhẹ và compare avg_latency_ms với target 35ms.

Dùng cho:
- Local kiểm tra nhanh trước khi commit
- CI gate: nếu benchmark fail, có thể block PR (tùy cấu hình)

Usage:
    python scripts/check_benchmark_target.py              # check default 35ms target
    python scripts/check_benchmark_target.py --target 50  # custom target
    python scripts/check_benchmark_target.py --threshold 5.0  # max delta before warn

Exit codes:
    0 — PASS: avg_latency <= target
    1 — FAIL: avg_latency > target
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_SCRIPT = ROOT / "scripts" / "benchmark_inference.py"
DEFAULT_TARGET_MS = 35.0


def run_benchmark() -> dict:
    """Run benchmark_inference.py và parse JSON output."""
    try:
        result = subprocess.run(
            [sys.executable, str(BENCHMARK_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            print(f"[ERROR] benchmark Inference script failed (exit {result.returncode})")
            print(result.stderr[-500:] if result.stderr else "")
            sys.exit(2)

        # Output là JSON khi chạy thành công
        output = result.stdout
        # Tìm JSON object trong output (nằm giữa các dấu =====)
        start = output.find("{")
        end = output.rfind("}") + 1
        if start < 0 or end <= start:
            print("[ERROR] Không thể parse benchmark JSON output")
            print(output[-500:])
            sys.exit(2)

        return json.loads(output[start:end])
    except subprocess.TimeoutExpired:
        print("[ERROR] Benchmark timed out after 120s")
        sys.exit(2)
    except FileNotFoundError:
        print(f"[ERROR] Benchmark script not found: {BENCHMARK_SCRIPT}")
        sys.exit(2)


def check_target(benchmark: dict, target_ms: float, warning_delta: float) -> int:
    """So sánh avg_latency với target. Return 0 = pass, 1 = fail."""
    avg_ms = benchmark.get("avg_latency_ms")
    if avg_ms is None:
        print("[ERROR] benchmark result thiếu avg_latency_ms")
        return 2

    fps = benchmark.get("fps", 0)
    diff_ms = avg_ms - target_ms
    status = "PASS" if diff_ms <= 0 else "FAIL"
    exit_code = 0 if diff_ms <= 0 else 1

    print("=" * 60)
    print("ApexInspect AI — Benchmark Target Check")
    print("=" * 60)
    print(f"  Model:          {benchmark.get('model')}")
    print(f"  Hardware:       {benchmark.get('hardware')}")
    print(f"  ONNX Runtime:  {benchmark.get('onnx_runtime')}")
    print(f"  Target:         {target_ms:.1f} ms ({1000/target_ms:.1f} FPS)")
    print(f"  Measured avg:   {avg_ms:.2f} ms ({fps:.1f} FPS)")
    print(f"  Delta:          {diff_ms:+.2f} ms vs target")
    print(f"  Status:         {status}")
    print("-" * 60)

    if warning_delta > 0 and diff_ms > 0 and diff_ms <= warning_delta:
        print(f"  WARNING: Within warning threshold ({warning_delta}ms above target)")
        print(f"           Consider optimizing before production deployment.")
        # Vẫn exit 0 nếu chỉ warning (không block CI)
        return 0

    if exit_code == 1:
        print(f"  RESULT: FAIL — measured {avg_ms:.2f}ms > target {target_ms:.1f}ms")
        print(f"          Runtime target: ≤{target_ms}ms trên cloud 2-vCPU")
        print(f"          Local macOS measurement: {avg_ms:.2f}ms ({fps:.1f} FPS)")
        print(f"          Lưu ý: benchmark này chạy trên local, chưa phải cloud thật.")
        print(f"          Production performance có thể khác.")
    else:
        print(f"  RESULT: PASS — measured {avg_ms:.2f}ms ≤ target {target_ms:.1f}ms")

    print("=" * 60)
    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="Check ApexInspect AI inference benchmark against target latency."
    )
    parser.add_argument(
        "--target", "-t",
        type=float,
        default=DEFAULT_TARGET_MS,
        help=f"Target latency in ms (default: {DEFAULT_TARGET_MS})",
    )
    parser.add_argument(
        "--warn-delta", "-w",
        type=float,
        default=5.0,
        help="Delta ms above target để warning nhưng không fail (default: 5.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save benchmark result JSON (besides benchmark_results.json)",
    )
    args = parser.parse_args()

    print(f"\nRunning benchmark (target={args.target}ms, warn_delta={args.warn_delta}ms)...\n")
    benchmark = run_benchmark()

    # Lưu kết quả
    benchmark_path = ROOT / "benchmark_results.json"
    benchmark_path.write_text(json.dumps(benchmark, indent=2) + "\n")
    print(f"Saved benchmark to: {benchmark_path}")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(benchmark, indent=2) + "\n")
        print(f"Saved copy to: {out_path}")

    exit_code = check_target(benchmark, args.target, args.warn_delta)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()