"""Optional Triton implementations of SACCADE's bounded hot paths.

Triton is deliberately optional: importing SACCADE never requires a CUDA
installation.  The wrappers below only dispatch to a kernel for CUDA tensors
when the operation is inference-only.  Training uses the PyTorch reference
path, which gives PyTorch a complete and well-tested autograd graph (and avoids
silently returning fake benchmark numbers on machines without Triton).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

try:  # pragma: no cover - exercised only on CUDA/Triton runners
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - normal on CPU-only installations
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:  # keep CPU imports safe when Triton is absent
    @triton.jit
    def _ema_kernel(x, alpha, out, stride_b, stride_t, stride_d, T: tl.constexpr,
                    D: tl.constexpr, BLOCK_D: tl.constexpr):
        b = tl.program_id(0)
        d = tl.arange(0, BLOCK_D)
        running = tl.zeros((BLOCK_D,), dtype=tl.float32)
        first = tl.load(x + b * stride_b + d * stride_d, mask=d < D, other=0).to(tl.float32)
        for t in range(0, T):
            prev = tl.load(x + b * stride_b + (t - 1) * stride_t + d * stride_d,
                           mask=(d < D) & (t > 0), other=0).to(tl.float32)
            # The reference defines ema[t] from x[t-1], with ema[0] = x[0].
            running = tl.where(t == 0, first, alpha * running + (1 - alpha) * prev)
            tl.store(out + b * stride_b + t * stride_t + d * stride_d,
                     running.to(out.dtype.element_ty), mask=d < D)

    @triton.jit
    def _slot_score_kernel(state, keys, scores, stride_sb, stride_sd,
                           stride_kb, stride_kn, stride_kd, stride_ob,
                           stride_on, N: tl.constexpr, D: tl.constexpr,
                           BLOCK_D: tl.constexpr):
        b = tl.program_id(0)
        n = tl.program_id(1)
        d = tl.arange(0, BLOCK_D)
        q = tl.load(state + b * stride_sb + d * stride_sd, mask=d < D, other=0).to(tl.float32)
        k = tl.load(keys + b * stride_kb + n * stride_kn + d * stride_kd,
                    mask=d < D, other=0).to(tl.float32)
        value = tl.sum(q * k, axis=0) / tl.sqrt(float(D))
        tl.store(scores + b * stride_ob + n * stride_on, value)

    @triton.jit
    def _cross_entropy_kernel(logits, targets, losses, stride_r, V: tl.constexpr,
                              BLOCK_V: tl.constexpr, ignore_index: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_V)
        values = tl.load(logits + row * stride_r + cols, mask=cols < V, other=-float("inf"))
        target = tl.load(targets + row)
        maximum = tl.max(values, axis=0)
        denominator = tl.sum(tl.exp(values - maximum), axis=0)
        selected = tl.load(logits + row * stride_r + target, mask=target != ignore_index, other=0.0)
        loss = tl.where(target == ignore_index, 0.0,
                        maximum + tl.log(denominator) - selected)
        tl.store(losses + row, loss)

    @triton.jit
    def _local_attention_kernel(q, k, v, out, sqb, sqh, sqt, sqd,
                                skb, skh, skt, skd, svb, svh, svt, svd,
                                sob, soh, sot, sod, T: tl.constexpr,
                                D: tl.constexpr, W: tl.constexpr,
                                BLOCK_D: tl.constexpr):
        pid = tl.program_id(0)
        h = pid % tl.num_programs(1)
        tmp = pid // tl.num_programs(1)
        t = tmp % T
        b = tmp // T
        d = tl.arange(0, BLOCK_D)
        qv = tl.load(q + b * sqb + h * sqh + t * sqt + d * sqd,
                     mask=d < D, other=0).to(tl.float32)
        maximum = -float("inf")
        for j in range(0, W):
            pos = t - W + 1 + j
            valid = (pos >= 0) & (pos <= t)
            kv = tl.load(k + b * skb + h * skh + pos * skt + d * skd,
                         mask=(d < D) & valid, other=0).to(tl.float32)
            score = tl.sum(qv * kv, axis=0) / tl.sqrt(float(D))
            maximum = tl.maximum(maximum, tl.where(valid, score, -float("inf")))
        denominator = 0.0
        for j in range(0, W):
            pos = t - W + 1 + j
            valid = (pos >= 0) & (pos <= t)
            kv = tl.load(k + b * skb + h * skh + pos * skt + d * skd,
                         mask=(d < D) & valid, other=0).to(tl.float32)
            score = tl.sum(qv * kv, axis=0) / tl.sqrt(float(D))
            denominator += tl.where(valid, tl.exp(score - maximum), 0.0)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for j in range(0, W):
            pos = t - W + 1 + j
            valid = (pos >= 0) & (pos <= t)
            kv = tl.load(k + b * skb + h * skh + pos * skt + d * skd,
                         mask=(d < D) & valid, other=0).to(tl.float32)
            vv = tl.load(v + b * svb + h * svh + pos * svt + d * svd,
                         mask=(d < D) & valid, other=0).to(tl.float32)
            score = tl.sum(qv * kv, axis=0) / tl.sqrt(float(D))
            acc += tl.where(valid, tl.exp(score - maximum) / denominator, 0.0) * vv
        tl.store(out + b * sob + h * soh + t * sot + d * sod,
                 acc.to(out.dtype.element_ty), mask=d < D)


def fused_ema_update(x: Tensor, alpha: Tensor | float) -> Tensor:
    """Compute the sequential EMA with an optional Triton inference kernel."""
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, time, features]")
    a = float(alpha.detach().item()) if isinstance(alpha, Tensor) else float(alpha)
    if not (triton_available() and x.is_cuda and not torch.is_grad_enabled()
            and x.is_contiguous()):
        values = [x[:, 0]]
        for i in range(1, x.size(1)):
            values.append(a * values[-1] + (1 - a) * x[:, i - 1])
        return torch.stack(values, dim=1) if values else x.new_empty(x.shape)
    out = torch.empty_like(x)
    block = triton.next_power_of_2(x.size(-1))
    _ema_kernel[(x.size(0),)](x, a, out, x.stride(0), x.stride(1), x.stride(2),
                              T=x.size(1), D=x.size(2), BLOCK_D=block)
    return out


def slot_scores(state: Tensor, keys: Tensor) -> Tensor:
    """Dot-product slot scores; top-k remains torch.topk for stable semantics."""
    if state.ndim != 2 or keys.ndim != 3 or state.shape[:1] != keys.shape[:1]:
        raise ValueError("state must be [B,D] and keys [B,N,D]")
    if not (triton_available() and state.is_cuda and not torch.is_grad_enabled()
            and state.is_contiguous() and keys.is_contiguous()):
        return torch.bmm(state.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) / math.sqrt(keys.size(-1))
    out = torch.empty(state.size(0), keys.size(1), device=state.device, dtype=torch.float32)
    block = triton.next_power_of_2(state.size(-1))
    _slot_score_kernel[(state.size(0), keys.size(1))](
        state, keys, out, state.stride(0), state.stride(1), keys.stride(0),
        keys.stride(1), keys.stride(2), out.stride(0), out.stride(1),
        N=keys.size(1), D=keys.size(-1), BLOCK_D=block)
    return out.to(dtype=state.dtype)


def mtp_cross_entropy(logits: Tensor, targets: Tensor, ignore_index: int = 0) -> Tensor:
    """Per-row cross entropy with an optional Triton inference fast path."""
    if logits.ndim != 2 or targets.ndim != 1 or logits.size(0) != targets.size(0):
        raise ValueError("logits must be [N,V] and targets must be [N]")
    if not (triton_available() and logits.is_cuda and not torch.is_grad_enabled()
            and logits.is_contiguous() and targets.is_contiguous()):
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)
    out = torch.empty(logits.size(0), device=logits.device, dtype=torch.float32)
    block = triton.next_power_of_2(logits.size(1))
    _cross_entropy_kernel[(logits.size(0),)](
        logits, targets, out, logits.stride(0), V=logits.size(1),
        BLOCK_V=block, ignore_index=ignore_index)
    valid = targets != ignore_index
    return out[valid].mean() if valid.any() else out.new_zeros(())


def causal_local_attention(q: Tensor, k: Tensor, v: Tensor, window: int) -> Tensor:
    """Reference-compatible causal local attention wrapper.

    The compact Triton kernel is intentionally kept as a separate opt-in
    primitive; the module-level attention continues using PyTorch projections.
    """
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k and v must have shape [B,H,T,D]")
    b, h, t, d = q.shape
    if not (triton_available() and q.is_cuda and not torch.is_grad_enabled()
            and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        idx = torch.arange(t, device=q.device)
        mask = (idx[None, :] > idx[:, None]) | (idx[None, :] < idx[:, None] - window + 1)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
        logits = logits.masked_fill(mask, float("-inf"))
        return torch.matmul(torch.softmax(logits, -1), v)
    out = torch.empty_like(q)
    block = triton.next_power_of_2(d)
    _local_attention_kernel[(b * t, h)](
        q, k, v, out, *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        T=t, D=d, W=window, BLOCK_D=block, num_warps=4)
    return out
