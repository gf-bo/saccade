from dataclasses import dataclass
import torch
from torch import Tensor, nn
from .config import SaccadeConfig
from .blocks import AbsoluteEmbedding, CausalLocalAttention, DynamicChunker, MiniSSM, SlotMemory, SlotRouter, FineAttention
from .streaming import StreamingState

@dataclass
class SaccadeOutput:
    logits: Tensor
    boundary_probs: Tensor
    boundaries: Tensor
    coarse_boundaries: Tensor
    route_scores: Tensor
    route_indices: Tensor
    ssm_logits: Tensor
    state: StreamingState

class SACCADE(nn.Module):
    def __init__(self, config: SaccadeConfig):
        super().__init__()
        self.config = config
        self.embed = AbsoluteEmbedding(config)
        self.local = CausalLocalAttention(config)
        self.chunker = DynamicChunker(config)
        self.ssm = MiniSSM(config)
        self.memory = SlotMemory(config)
        self.router = SlotRouter(config)
        self.fine = FineAttention(config)
        self.ssm_to_model = nn.Linear(config.d_ssm, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.ssm_head = nn.Linear(config.d_ssm, config.vocab_size, bias=False)
        self.sequence_retrieval_gate = nn.Parameter(torch.tensor(-1.5))

    def forward_sequence(self, tokens: Tensor, state: StreamingState | None = None,
                         **kwargs) -> tuple[Tensor, StreamingState]:
        """Return causal logits for every input position in one batched pass.

        Embedding, local attention and the SSM are evaluated once for the
        complete teacher-forced sequence.  Memory writes remain incremental,
        so a position can only route to slots written at or before itself.
        This is substantially cheaper than invoking ``forward`` once per
        token and preserves the exact streaming state at the end.
        """
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, time]")
        batch, length = tokens.shape
        if length == 0:
            empty = tokens.new_empty(tokens.size(0), 0, self.config.vocab_size,
                                     dtype=self.embed.embedding.weight.dtype)
            return empty, state or StreamingState.empty(batch, self.config.d_ssm, tokens.device)

        h, addr = self.embed(tokens)
        local = self.local(h, self.config.w_fine)
        mixed, _, _, coarse = self.chunker(local, kwargs.get("threshold"))
        current_ssm = (None if state is None else
                       state.ssm.to(device=tokens.device, dtype=mixed.dtype))
        ssm_y, final_ssm = self.ssm(mixed, current_ssm)
        direct = self.lm_head(local + self.ssm_to_model(ssm_y))

        base = (state.next_address.to(tokens.device) if state is not None else
                torch.zeros(batch, dtype=torch.long, device=tokens.device))
        slots = None if state is None else [
            [type(slot)(slot.start, slot.end,
                        slot.embedding.to(tokens.device, mixed.dtype),
                        slot.tokens.to(tokens.device, mixed.dtype))
             for slot in row] for row in state.slots]
        logits = []
        # Chunking this loop bounds temporary context lists and makes the
        # teacher-forcing path friendly to long streams without changing
        # attention locality.
        chunk_size = max(1, self.config.sequence_chunk_size)
        route_temperature = kwargs.get("temperature", 1.0)
        route_top_k = kwargs.get("top_k")
        for chunk_start in range(0, length, chunk_size):
            chunk_end = min(length, chunk_start + chunk_size)
            for t in range(chunk_start, chunk_end):
                one_addr = addr[:, t:t + 1] + base.unsqueeze(1)
                slots = self.memory.write(mixed[:, t:t + 1], one_addr,
                                          coarse[:, t:t + 1], slots)
                scores, indices = self.router(ssm_y[:, t], slots,
                                              route_temperature, route_top_k)
                rows = []
                for b in range(batch):
                    pieces = [slots[b][int(i)].tokens[-self.config.l_slot:]
                              for i in indices[b] if 0 <= int(i) < len(slots[b])]
                    pieces.append(mixed[b, max(0, t - self.config.local_window + 1):t + 1])
                    pieces.append(self.ssm_to_model(ssm_y[b, t]).unsqueeze(0))
                    context = torch.cat(pieces, 0)
                    rows.append(self.fine(mixed[b, t], context))
                retrieval = torch.stack(rows)
                logits.append(direct[:, t] +
                              self.sequence_retrieval_gate.sigmoid() * self.lm_head(retrieval))
        next_address = base + length
        return torch.stack(logits, dim=1), StreamingState(final_ssm, slots, next_address)
    def forward(self, tokens: Tensor, state: StreamingState | None = None, temperature: float = 1.0,
                threshold: float | None = None, top_k: int | None = None) -> SaccadeOutput:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, time]")
        batch, length = tokens.shape
        if state is not None and (state.ssm.shape[0] != batch or len(state.slots) != batch):
            raise ValueError("streaming state batch does not match tokens")
        if length == 0:
            device = tokens.device
            s = state.ssm.to(device=device, dtype=self.embed.embedding.weight.dtype) if state is not None else torch.zeros(batch, self.config.d_ssm, device=device, dtype=self.embed.embedding.weight.dtype)
            slots = state.slots if state is not None else [[] for _ in range(batch)]
            scores, indices = self.router(s, slots, temperature, top_k)
            z = torch.zeros(batch, self.config.vocab_size, device=device, dtype=s.dtype)
            empty = torch.empty(batch, 0, device=device, dtype=s.dtype)
            return SaccadeOutput(z, empty, empty, empty, scores, indices, self.ssm_head(s), StreamingState(s, slots, (state.next_address if state is not None else torch.zeros(batch, dtype=torch.long, device=device))))
        h, addr = self.embed(tokens)
        local = self.local(h, self.config.w_fine)
        mixed, p, boundaries, coarse = self.chunker(local, threshold)
        state_ssm = None if state is None else state.ssm.to(device=tokens.device, dtype=mixed.dtype)
        if state is not None:
            # Slot tensors are cached activations, so normalize them when a
            # state is moved between devices or precision modes.
            state_slots = [[type(slot)(slot.start, slot.end,
                                        slot.embedding.to(tokens.device, mixed.dtype),
                                        slot.tokens.to(tokens.device, mixed.dtype))
                            for slot in row] for row in state.slots]
        else:
            state_slots = None
        ssm_y, s = self.ssm(mixed, state_ssm)
        base = state.next_address.to(device=tokens.device) if state is not None else torch.zeros(batch, dtype=torch.long, device=tokens.device)
        addr = addr + base.unsqueeze(1)
        slots = self.memory.write(mixed, addr, coarse, state_slots)
        scores, indices = self.router(s, slots, temperature, top_k)
        contexts = []
        for b in range(tokens.size(0)):
            selected = indices[b] if indices.numel() else []
            pieces = [slots[b][int(i)].tokens[-self.config.l_slot:] for i in selected if int(i) >= 0 and int(i) < len(slots[b])]
            pieces.append(mixed[b, -self.config.local_window:])
            pieces.append(self.ssm_to_model(ssm_y[b, -1]).unsqueeze(0))
            contexts.append(torch.cat(pieces, 0))
        out = torch.stack([self.fine(mixed[b, -1], contexts[b]) for b in range(tokens.size(0))])
        new_state = StreamingState(s, slots, base + length)
        return SaccadeOutput(self.lm_head(out), p, boundaries, coarse, scores, indices, self.ssm_head(s), new_state)
