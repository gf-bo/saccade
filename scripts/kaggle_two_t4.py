"""Kaggle entry point for two T4s.

Run with internet disabled and the repository attached as a Kaggle dataset:
```
torchrun --standalone --nproc_per_node=2 scripts/kaggle_two_t4.py``
The script also works on one GPU and explicitly reports the selected device.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saccade import SACCADE, SaccadeConfig
from saccade.benchmark import generate_retrieval_datasets, _batch
from saccade.triton_kernels import fused_ema_update, slot_scores, triton_available


def _ema_reference(x: torch.Tensor, alpha: float) -> torch.Tensor:
    values = [x[:, 0]]
    for i in range(1, x.size(1)):
        values.append(alpha * values[-1] + (1 - alpha) * x[:, i - 1])
    return torch.stack(values, dim=1)


def _slot_reference(state: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    return torch.bmm(state.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) / keys.size(-1) ** 0.5


def _measure(fn, device: torch.device, warmup: int = 10, iters: int = 100) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000 / iters


def _measure_peak(fn, device: torch.device) -> float:
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 2**20


def _kernel_report(device: torch.device) -> dict:
    x = torch.randn(8, 1024, 128, device=device)
    state = torch.randn(8, 128, device=device)
    keys = torch.randn(8, 64, 128, device=device)
    alpha = 0.9
    ema_ref = _ema_reference(x, alpha)
    slots_ref = _slot_reference(state, keys)
    with torch.inference_mode():
        ema_fast = fused_ema_update(x, alpha)
        slots_fast = slot_scores(state, keys)
    report = {
        "triton": triton_available(),
        "ema_pytorch_ms": _measure(lambda: _ema_reference(x, alpha), device),
        "ema_triton_ms": _measure(lambda: fused_ema_update(x, alpha), device),
        "slot_scores_pytorch_ms": _measure(lambda: _slot_reference(state, keys), device),
        "slot_scores_triton_ms": _measure(lambda: slot_scores(state, keys), device),
        "ema_max_abs_error": float((ema_fast - ema_ref).abs().max()),
        "slot_scores_max_abs_error": float((slots_fast - slots_ref).abs().max()),
        "ema_pytorch_peak_mb": _measure_peak(lambda: _ema_reference(x, alpha), device),
        "ema_triton_peak_mb": _measure_peak(lambda: fused_ema_update(x, alpha), device),
        "slot_scores_pytorch_peak_mb": _measure_peak(lambda: _slot_reference(state, keys), device),
        "slot_scores_triton_peak_mb": _measure_peak(lambda: slot_scores(state, keys), device),
    }
    report["ema_speedup"] = report["ema_pytorch_ms"] / report["ema_triton_ms"]
    report["slot_scores_speedup"] = report["slot_scores_pytorch_ms"] / report["slot_scores_triton_ms"]
    report["ema_elements_per_second"] = x.numel() / (report["ema_triton_ms"] / 1000)
    report["slot_scores_elements_per_second"] = keys.numel() / (report["slot_scores_triton_ms"] / 1000)
    return report


def main() -> None:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world > 1
    if not torch.cuda.is_available():
        raise SystemExit("Kaggle two-T4 script requires CUDA; no GPU was used.")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        dist.init_process_group("nccl")
    model = SACCADE(SaccadeConfig(
        vocab_size=64, d_model=40, d_addr=10, d_ssm=20, n_heads=4,
        w_fine=8, w_coarse=32, s_max=8, l_slot=64, top_k=2,
        local_window=8, sequence_chunk_size=64)).to(device)
    model = DDP(model, device_ids=[local_rank]) if distributed else model
    datasets = generate_retrieval_datasets(1000, 512, 1729)
    train = datasets["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    steps = 8
    last_loss = None
    for step in range(steps):
        start = (rank * steps + step) % max(1, len(train) - 1)
        batch = _batch(train, [start], str(device))
        optimizer.zero_grad(set_to_none=True)
        logits = model.module.forward_sequence(batch["input_ids"])[0] if distributed else model.forward_sequence(batch["input_ids"])[0]
        loss = criterion(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = loss.detach()
    benchmark = _kernel_report(device)
    if rank == 0:
        print({"world_size": world, "devices": [torch.cuda.get_device_name(i) for i in range(world)],
               "loss": float(last_loss), "kernel_benchmark": benchmark})
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
