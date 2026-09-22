from dataclasses import dataclass

@dataclass
class SaccadeConfig:
    vocab_size: int = 32000
    d_model: int = 256
    d_addr: int = 16
    d_ssm: int = 64
    n_heads: int = 4
    w_fine: int = 32
    w_coarse: int = 256
    s_max: int = 128
    l_slot: int = 512
    top_k: int = 2
    local_window: int = 64
    boundary_threshold: float = 0.5
    ema_alpha: float = 0.95
    lambda_balance: float = 0.01
    lambda_chunk: float = 0.01
    lambda_ssm: float = 0.01
    lambda_router_entropy: float = 0.001
    ffn_mult: int = 4
    sequence_chunk_size: int = 64
    pad_id: int = 0

    def __post_init__(self) -> None:
        for name in ("d_model", "d_addr", "d_ssm", "n_heads", "w_fine", "w_coarse", "s_max", "l_slot", "top_k", "ffn_mult", "sequence_chunk_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 1 <= self.top_k <= self.s_max:
            raise ValueError("top_k must be in [1, s_max]")

@dataclass
class CurriculumConfig:
    warmup_fraction: float = 0.1
    transition_fraction: float = 0.3
    initial_k: int | None = None
    initial_threshold: float = 0.1
    final_temperature: float = 1.0
    initial_temperature: float = 2.0
    clip_norm: float = 1.0

    def values(self, step: int, total_steps: int, target_k: int, target_threshold: float) -> tuple[int, float, float]:
        progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
        warm = self.initial_k or target_k
        if progress < self.warmup_fraction:
            return warm, self.initial_threshold, self.initial_temperature
        transition = min(max((progress - self.warmup_fraction) / max(self.transition_fraction, 1e-6), 0.0), 1.0)
        k = round(warm + (target_k - warm) * transition)
        threshold = self.initial_threshold + (target_threshold - self.initial_threshold) * transition
        temperature = self.initial_temperature + (self.final_temperature - self.initial_temperature) * transition
        return max(1, k), threshold, temperature
