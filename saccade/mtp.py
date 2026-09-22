"""Multi-token prediction utilities for SACCADE."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from .triton_kernels import mtp_cross_entropy


@dataclass
class MTPOutput:
    """Teacher-forced outputs for the primary and speculative MTP heads."""

    logits: Tensor
    mtp_logits: tuple[Tensor, ...]
    state: object


def mtp_loss(
    logits: Tensor,
    mtp_logits: tuple[Tensor, ...],
    tokens: Tensor,
    pad_id: int = 0,
    weight: float = 0.5,
) -> dict[str, Tensor]:
    """Compute primary next-token and auxiliary future-token losses.

    ``mtp_logits[i]`` predicts the token at offset ``i + 2`` from each
    teacher-forced position. Invalid positions at the end are ignored.
    """
    if logits.ndim != 3 or tokens.ndim != 2:
        raise ValueError("logits must be [B,T,V] and tokens must be [B,T]")
    primary_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    primary_targets = tokens[:, 1:].reshape(-1)
    primary = (mtp_cross_entropy(primary_logits, primary_targets, pad_id)
               if not torch.is_grad_enabled() else
               F.cross_entropy(primary_logits, primary_targets, ignore_index=pad_id))
    auxiliary = primary.new_zeros(())
    terms = 0
    for offset, future in enumerate(mtp_logits, start=2):
        usable = min(future.size(1), max(0, tokens.size(1) - offset))
        if usable:
            future_logits = future[:, :usable].reshape(-1, future.size(-1))
            future_targets = tokens[:, offset:offset + usable].reshape(-1)
            term = (mtp_cross_entropy(future_logits, future_targets, pad_id)
                    if not torch.is_grad_enabled() else
                    F.cross_entropy(future_logits, future_targets, ignore_index=pad_id))
            auxiliary = auxiliary + term
            terms += 1
    if terms:
        auxiliary = auxiliary / terms
    total = primary + weight * auxiliary
    return {"loss": total, "lm": primary, "mtp": auxiliary}


def mtp_targets(tokens: Tensor, horizon: int, pad_id: int = 0) -> Tensor:
    """Return stacked future-token targets, padded at the sequence tail."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    batch, length = tokens.shape
    result = tokens.new_full((batch, length, horizon), pad_id)
    for i in range(horizon):
        if i + 2 < length:
            result[:, :length - i - 2, i] = tokens[:, i + 2:]
    return result
