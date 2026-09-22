"""SACCADE: local, addressable, causal memory for PyTorch language models."""

__version__ = "0.3.0"

from .config import SaccadeConfig, CurriculumConfig
from .model import SACCADE, SaccadeOutput
from .streaming import StreamingState
from .losses import saccade_loss
from .training import train_step, TrainStepResult, SaccadeTrainer
from .mtp import MTPOutput, mtp_loss, mtp_targets
from .factory import create_model, create_trainer
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
    "SaccadeTrainer",
    "MTPOutput",
    "mtp_loss",
    "mtp_targets",
    "create_model",
    "create_trainer",
    "triton_available",
]
