from dataclasses import dataclass
import torch
from torch import Tensor
from .config import CurriculumConfig
from .losses import saccade_loss
from .model import SACCADE
from .mtp import mtp_loss

@dataclass
class TrainStepResult:
    losses: dict[str, Tensor]
    grad_norm: Tensor
    k: int
    threshold: float
    temperature: float
    mtp_enabled: bool = False

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


class SaccadeTrainer:
    """Reusable training loop for standard or multi-token prediction."""

    def __init__(self, model: SACCADE, optimizer: torch.optim.Optimizer,
                 curriculum: CurriculumConfig | None = None):
        self.model = model
        self.optimizer = optimizer
        self.curriculum = curriculum or CurriculumConfig()

    def step(self, tokens: Tensor, step: int, total_steps: int) -> TrainStepResult:
        k, threshold, temperature = self.curriculum.values(
            step, total_steps, self.model.config.top_k,
            self.model.config.boundary_threshold)
        old_k = self.model.config.top_k
        self.model.config.top_k = min(k, self.model.config.s_max)
        try:
            if self.model.config.mtp_num_tokens:
                output = self.model.forward_sequence_mtp(
                    tokens, threshold=threshold, temperature=temperature,
                    top_k=self.model.config.top_k)
                losses = mtp_loss(
                    output.logits, output.mtp_logits, tokens,
                    self.model.config.pad_id, self.model.config.mtp_loss_weight)
            else:
                logits, _ = self.model.forward_sequence(
                    tokens, threshold=threshold, temperature=temperature,
                    top_k=self.model.config.top_k)
                losses = {"loss": torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tokens[:, 1:].reshape(-1),
                    ignore_index=self.model.config.pad_id)}
        finally:
            self.model.config.top_k = old_k
        self.optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.curriculum.clip_norm)
        self.optimizer.step()
        return TrainStepResult(
            losses, norm, k, threshold, temperature,
            bool(self.model.config.mtp_num_tokens))

    def fit(self, batches, steps: int | None = None) -> list[TrainStepResult]:
        """Train over an iterable of token batches and return step records."""
        history: list[TrainStepResult] = []
        limit = steps if steps is not None else float("inf")
        for step, tokens in enumerate(batches):
            if step >= limit:
                break
            history.append(self.step(tokens, step, steps or 1))
        return history
