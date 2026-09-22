"""SACCADE: local, addressable, causal memory for PyTorch language models."""

__version__ = "0.2.0"

from .config import SaccadeConfig, CurriculumConfig
from .model import SACCADE, SaccadeOutput
from .streaming import StreamingState
from .losses import saccade_loss
from .training import train_step, TrainStepResult
from .triton_kernels import triton_available

__all__ = [
    "__version__",
    "SaccadeConfig",
    "CurriculumConfig",
    "SACCADE",
    "SaccadeOutput",
    "StreamingState",
    "saccade_loss",
    "train_step",
    "TrainStepResult",
    "triton_available",
]
