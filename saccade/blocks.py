import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .config import SaccadeConfig
from .streaming import Slot
from .triton_kernels import fused_ema_update, slot_scores

class AbsoluteEmbedding(nn.Module):
    def __init__(self, cfg: SaccadeConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        h = self.norm(self.embedding(tokens))
        return h, torch.arange(tokens.size(1), device=tokens.device, dtype=torch.long).expand(tokens.size(0), -1)

class CausalLocalAttention(nn.Module):
    def __init__(self, cfg: SaccadeConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.norm = nn.LayerNorm(cfg.d_model)
        hidden = cfg.ffn_mult * cfg.d_model
        self.ffn_norm = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model),
        )
    def forward(self, x: Tensor, window: int) -> Tensor:
        n = x.size(1)
        mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        if window < n:
            mask |= torch.tril(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=-window)
        y, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm(x + y)
        return x + self.ffn(self.ffn_norm(x))

class DynamicChunker(nn.Module):
    def __init__(self, cfg: SaccadeConfig):
        super().__init__()
        self.coarse_window = cfg.w_coarse
        self.q, self.k = nn.Linear(cfg.d_model, cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model)
        self.alpha_logit = nn.Parameter(torch.tensor(3.0))
        self.threshold = nn.Parameter(torch.tensor(cfg.boundary_threshold).logit())
        self.ema_alpha = nn.Parameter(torch.tensor(cfg.ema_alpha).logit())
    def forward(self, x: Tensor, threshold: float | Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.size(1) == 0:
            empty = x.new_empty(x.size(0), 0)
            return x, empty, empty, empty
        q, k = self.q(x), self.k(x)
        sim = torch.nn.functional.cosine_similarity(q[:, 1:], k[:, :-1], dim=-1)
        p = torch.cat([torch.ones(x.size(0), 1, device=x.device), 0.5 * (1 - sim)], dim=1)
        t = self.threshold.sigmoid() if threshold is None else torch.as_tensor(threshold, device=x.device, dtype=x.dtype)
        hard = (p >= t).to(x.dtype)
        # Straight-through estimator: forward is a hard boundary, backward
        # follows a sigmoid whose argument includes the learnable threshold.
        soft_boundary = torch.sigmoid((p - t) / 0.1)
        boundary = hard + soft_boundary - soft_boundary.detach()
        hard[:, 0] = 1
        alpha = self.ema_alpha.sigmoid()
        gate = self.alpha_logit.sigmoid()
        ema = fused_ema_update(x, alpha)
        smooth = gate * (p.unsqueeze(-1) * x + (1 - p.unsqueeze(-1)) * ema) + (1 - gate) * x
        gated = hard.unsqueeze(-1) * x + (1 - hard.unsqueeze(-1)) * ema
        mixed = smooth + (gated - smooth).detach()
        coarse = boundary.clone()
        if x.size(1) > 1:
            coarse[:, 1:] = boundary[:, 1:] * (torch.arange(x.size(1)-1, device=x.device) % max(1, self.coarse_window) == 0)
        return mixed, p, boundary, coarse

class MiniSSM(nn.Module):
    def __init__(self, cfg: SaccadeConfig):
        super().__init__()
        self.in_proj = nn.Linear(cfg.d_model, cfg.d_ssm)
        self.delta = nn.Linear(cfg.d_model, cfg.d_ssm)
        self.b = nn.Linear(cfg.d_model, cfg.d_ssm)
        self.c = nn.Linear(cfg.d_model, cfg.d_ssm)
        self.log_a = nn.Parameter(torch.zeros(cfg.d_ssm))
    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        s = torch.zeros(x.size(0), self.log_a.numel(), device=x.device, dtype=x.dtype) if state is None else state
        ys = []
        for i in range(x.size(1)):
            u = x[:, i]
            # log_a is a learned positive decay-rate scale (used in the ZOH-like update).
            a = torch.exp(-F.softplus(self.delta(u)) * torch.exp(self.log_a).to(u))
            s = a * s + self.b(u) * self.in_proj(u)
            ys.append(self.c(u) * s)
        return torch.stack(ys, 1), s

class SlotMemory(nn.Module):
    def __init__(self, cfg: SaccadeConfig):
        super().__init__()
        self.compress = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_addr))
        self.cfg = cfg
    def write(self, x: Tensor, addresses: Tensor, boundaries: Tensor, prior: list[list[Slot]] | None = None) -> list[list[Slot]]:
        slots_batch = [list(row) for row in (prior or [[] for _ in range(x.size(0))])]
        if len(slots_batch) != x.size(0):
            raise ValueError("prior slots must contain one list per batch item")
        for b in range(x.size(0)):
            slots = slots_batch[b]
            start = 0
            for t in range(x.size(1)):
                if t == x.size(1)-1 or boundaries[b, t].item() > 0.5:
                    for lo in range(start, t + 1, self.cfg.l_slot):
                        hi = min(lo + self.cfg.l_slot, t + 1)
                        seg = x[b, lo:hi]
                        emb = self.compress(seg.mean(0))
                        slots.append(Slot(int(addresses[b, lo]), int(addresses[b, hi - 1]) + 1, emb, seg))
                    start = t + 1
            while len(slots) > self.cfg.s_max:
                first, second = slots.pop(0), slots.pop(0)
                first_len = max(0, first.end - first.start)
                second_len = max(0, second.end - second.start)
                total = max(1, first_len + second_len)
                emb = (first.embedding * first_len + second.embedding * second_len) / total
                tokens = torch.cat([first.tokens, second.tokens], 0)[-self.cfg.l_slot:]
                slots.insert(0, Slot(first.start, second.end, emb, tokens))
            slots_batch[b] = slots
        return slots_batch

class SlotRouter(nn.Module):
    def __init__(self, cfg: SaccadeConfig):
        super().__init__()
        self.query = nn.Linear(cfg.d_ssm, cfg.d_addr)
        self.top_k = cfg.top_k
    def forward(self, state: Tensor, slots: list[list[Slot]], temperature: float = 1.0, top_k: int | None = None) -> tuple[Tensor, Tensor]:
        n = max((len(row) for row in slots), default=0)
        k = top_k or self.top_k
        if not n:
            return state.new_full((state.size(0), k), float("-inf")), torch.full((state.size(0), k), -1, dtype=torch.long, device=state.device)
        keys = state.new_zeros(state.size(0), n, self.query.out_features)
        valid = torch.zeros(state.size(0), n, dtype=torch.bool, device=state.device)
        for b, row in enumerate(slots):
            if row:
                keys[b, :len(row)] = torch.stack([s.embedding.to(device=state.device, dtype=state.dtype) for s in row])
                valid[b, :len(row)] = True
        scores = slot_scores(self.query(state), keys) / max(temperature, 1e-6)
        scores = scores.masked_fill(~valid, float("-inf"))
        if n < k:
            scores = torch.cat([scores, state.new_full((state.size(0), k - n), float("-inf"))], dim=1)
            valid = torch.cat([valid, torch.zeros(state.size(0), k - n, dtype=torch.bool, device=state.device)], dim=1)
        top_scores, indices = scores.topk(k, dim=-1)
        indices = indices.masked_fill(~valid.gather(1, indices), -1)
        top_scores = top_scores.masked_fill(indices < 0, float("-inf"))
        return top_scores, indices

class FineAttention(nn.Module):
    def __init__(self, cfg: SaccadeConfig):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(cfg.d_model, cfg.d_model) for _ in range(4))
    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        q, k, v = self.q(query).unsqueeze(0).unsqueeze(1), self.k(context).unsqueeze(0), self.v(context).unsqueeze(0)
        return self.o(torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(k.size(-1)), -1) @ v).squeeze(0).squeeze(0)
