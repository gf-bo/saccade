from dataclasses import dataclass
import torch
from torch import Tensor
from .config import CurriculumConfig
from .losses import saccade_loss
from .model import SACCADE

@dataclass
class TrainStepResult:
    losses: dict[str, Tensor]
    grad_norm: Tensor
    k: int
    threshold: float
    temperature: float

def train_step(model: SACCADE, optimizer: torch.optim.Optimizer, tokens: Tensor, targets: Tensor,
               step: int, total_steps: int, curriculum: CurriculumConfig | None = None) -> TrainStepResult:
    """One CPU-friendly causal update; targets are the labels for the emitted token."""
    schedule = curriculum or CurriculumConfig()
    k, threshold, temperature = schedule.values(step, total_steps, model.config.top_k, model.config.boundary_threshold)
    old_k = model.config.top_k
    try:
        model.config.top_k = min(k, model.config.s_max)
        if targets.ndim == 2:
            # Efficient teacher forcing: compute all causal positions in one
            # batched feature pass instead of resetting local attention for
            # every token.
            logits, _ = model.forward_sequence(
                tokens, temperature=temperature, threshold=threshold,
                top_k=model.config.top_k)
            losses = {
                "loss": torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                    ignore_index=model.config.pad_id)
            }
        else:
            output = model(tokens, temperature=temperature, threshold=threshold, top_k=model.config.top_k)
            losses = saccade_loss(output.logits, targets, output.boundary_probs, output.route_scores,
                                  output.route_indices, output.ssm_logits, model.config)
    finally:
        model.config.top_k = old_k
    optimizer.zero_grad(set_to_none=True)
    losses["loss"].backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), schedule.clip_norm)
    optimizer.step()
    return TrainStepResult(losses, norm, k, threshold, temperature)
