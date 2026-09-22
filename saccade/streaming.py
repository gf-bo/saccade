from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class Slot:
    start: int
    end: int
    embedding: Tensor
    tokens: Tensor

@dataclass
class StreamingState:
    ssm: Tensor
    slots: list[list[Slot]]
    next_address: Tensor

    def __post_init__(self) -> None:
        if self.ssm.ndim != 2:
            raise ValueError("ssm must have shape [batch, d_ssm]")
        if len(self.slots) != self.ssm.size(0):
            raise ValueError("slots must contain one list per batch item")
        if self.next_address.ndim != 1 or self.next_address.size(0) != self.ssm.size(0):
            raise ValueError("next_address must have one value per batch item")

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "StreamingState":
        return StreamingState(
            self.ssm.to(device=device, dtype=dtype),
            [[Slot(x.start, x.end, x.embedding.to(device=device, dtype=dtype),
                   x.tokens.to(device=device, dtype=dtype)) for x in row] for row in self.slots],
            self.next_address.to(device=device),
        )

    @classmethod
    def empty(cls, batch: int, d_ssm: int, device: torch.device | None = None) -> "StreamingState":
        return cls(torch.zeros(batch, d_ssm, device=device), [[] for _ in range(batch)],
                   torch.zeros(batch, dtype=torch.long, device=device))

    def detach(self) -> "StreamingState":
        return StreamingState(
            self.ssm.detach(),
            [[Slot(x.start, x.end, x.embedding.detach(), x.tokens.detach()) for x in row] for row in self.slots],
            self.next_address.detach(),
        )
