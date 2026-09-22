import torch
from torch import Tensor
import torch.nn.functional as F
from .config import SaccadeConfig

def saccade_loss(logits: Tensor, targets: Tensor, boundary_probs: Tensor, route_scores: Tensor, route_indices: Tensor, ssm_logits: Tensor | None, cfg: SaccadeConfig, target_chunk: float | None = None) -> dict[str, Tensor]:
    lm = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=cfg.pad_id)
    if boundary_probs.size(1) <= 1:
        chunk = lm.new_zeros(())
    else:
        threshold = cfg.boundary_threshold
        lengths = []
        for row in boundary_probs:
            positions = torch.nonzero(row[1:] >= threshold, as_tuple=False).flatten() + 1
            ends = torch.cat([positions, row.new_tensor([row.numel()])])
            starts = torch.cat([row.new_tensor([0]), ends[:-1]])
            lengths.append((ends - starts).to(row.dtype))
        mean_len = torch.cat(lengths).mean()
        chunk = (mean_len - (target_chunk or cfg.w_coarse / 2)) ** 2
    if route_scores.numel():
        # Empty streaming memories return all ``-inf`` route scores.  Avoid
        # propagating NaNs through the auxiliary objective in that case.
        probs = torch.softmax(route_scores, -1).nan_to_num(0.0)
        valid = route_indices >= 0
        max_index = int(route_indices.clamp_min(0).max().item()) + 1 if route_indices.numel() else 1
        slot_count = max(cfg.s_max, max_index)
        chosen = torch.zeros(route_scores.size(0), slot_count, device=route_scores.device, dtype=probs.dtype)
        assigned = torch.zeros_like(chosen)
        safe_indices = route_indices.clamp_min(0)
        weights = valid.to(probs.dtype)
        chosen.scatter_add_(1, safe_indices, weights)
        assigned.scatter_add_(1, safe_indices, probs * weights)
        freq = chosen / chosen.sum(1, keepdim=True).clamp_min(1)
        mean_prob = assigned / assigned.sum(1, keepdim=True).clamp_min(1e-6)
        balance = slot_count * (freq * mean_prob).sum(-1).mean()
        # Penalize collapsed routing without rewarding arbitrarily sharp
        # distributions.  The small coefficient keeps this numerically benign.
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(-1).mean()
    else:
        balance = lm.new_zeros(())
        entropy = lm.new_zeros(())
    aux = lm.new_zeros(()) if ssm_logits is None or ssm_logits.numel() == 0 else F.cross_entropy(ssm_logits.reshape(-1, ssm_logits.size(-1)), targets.reshape(-1), ignore_index=cfg.pad_id)
    total = (lm + cfg.lambda_chunk * chunk + cfg.lambda_balance * balance
             + cfg.lambda_router_entropy * entropy + cfg.lambda_ssm * aux)
    return {"loss": total, "lm": lm, "chunk": chunk, "balance": balance,
            "router_entropy": entropy, "ssm": aux}
