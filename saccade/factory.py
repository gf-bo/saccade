"""Convenience constructors for end-to-end SACCADE experiments."""

from __future__ import annotations

import torch

from .config import CurriculumConfig, SaccadeConfig
from .model import SACCADE
from .training import SaccadeTrainer


def create_model(config: SaccadeConfig | None = None, **config_overrides) -> SACCADE:
    """Create a SACCADE model from a config or keyword overrides."""
    if config is not None and config_overrides:
        raise ValueError("pass either config or config_overrides, not both")
    return SACCADE(config or SaccadeConfig(**config_overrides))


def create_trainer(
    model: SACCADE,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    curriculum: CurriculumConfig | None = None,
) -> SaccadeTrainer:
    """Create an AdamW-backed trainer for standard LM or MTP training."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return SaccadeTrainer(model, optimizer, curriculum)
