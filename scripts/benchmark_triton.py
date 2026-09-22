"""Benchmark optional SACCADE kernels without inventing unavailable GPU data.

Examples:
  python scripts/benchmark_triton.py --device cuda
  python scripts/benchmark_triton.py --device cpu  # reference timing only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Allow ``python scripts\benchmark_triton.py`` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saccade.triton_kernels import fused_ema_update, slot_scores, triton_available


def ema_reference(x: torch.Tensor, alpha: float) -> torch.Tensor:
    values = [x[:, 0]]
    for i in range(1, x.size(1)):
        values.append(alpha * values[-1] + (1 - alpha) * x[:, i - 1])
    return torch.stack(values, dim=1)


def slot_scores_reference(state: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    return torch.bmm(state.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) / keys.size(-1) ** 0.5


def measure(fn, warmup: int, iters: int, device: torch.device) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def measure_memory(fn, device: torch.device) -> int:
    if device.type != "cuda":
        fn()
        return 0
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false; no GPU result was recorded.")
    device = torch.device(args.device)
    x = torch.randn(args.batch, args.length, args.dim, device=device)
    state = torch.randn(args.batch, args.dim, device=device)
    keys = torch.randn(args.batch, 64, args.dim, device=device)
    ema_ref = ema_reference(x, 0.9)
    slots_ref = slot_scores_reference(state, keys)
    with torch.inference_mode():
        ema_fast = fused_ema_update(x, 0.9)
        slots_fast = slot_scores(state, keys)
    result = {
        "device": str(device),
        "cuda": torch.cuda.is_available(),
        "triton": triton_available(),
        "ema_pytorch_ms": measure(lambda: ema_reference(x, 0.9), 5, args.iters, device),
        "ema_triton_ms": measure(lambda: fused_ema_update(x, 0.9), 5, args.iters, device),
        "slot_scores_pytorch_ms": measure(lambda: slot_scores_reference(state, keys), 5, args.iters, device),
        "slot_scores_triton_ms": measure(lambda: slot_scores(state, keys), 5, args.iters, device),
        "ema_max_abs_error": float((ema_fast - ema_ref).abs().max()),
        "slot_scores_max_abs_error": float((slots_fast - slots_ref).abs().max()),
        "ema_pytorch_peak_mb": measure_memory(lambda: ema_reference(x, 0.9), device) / 2**20,
        "ema_triton_peak_mb": measure_memory(lambda: fused_ema_update(x, 0.9), device) / 2**20,
        "slot_scores_pytorch_peak_mb": measure_memory(lambda: slot_scores_reference(state, keys), device) / 2**20,
        "slot_scores_triton_peak_mb": measure_memory(lambda: slot_scores(state, keys), device) / 2**20,
    }
    result["ema_speedup"] = result["ema_pytorch_ms"] / result["ema_triton_ms"]
    result["slot_scores_speedup"] = result["slot_scores_pytorch_ms"] / result["slot_scores_triton_ms"]
    result["ema_elements_per_second"] = args.batch * args.length * args.dim / (result["ema_triton_ms"] / 1000)
    result["slot_scores_elements_per_second"] = args.batch * 64 * args.dim / (result["slot_scores_triton_ms"] / 1000)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
