"""Training engine: trainer, optimizer/scheduler builders, metrics, regularizers."""

from __future__ import annotations

from vit.training.ema import ModelEma
from vit.training.metrics import AverageMeter, accuracy
from vit.training.mixup import MixupCutmix, SoftTargetCrossEntropy
from vit.training.optim import build_optimizer
from vit.training.scheduler import build_scheduler
from vit.training.trainer import Trainer, TrainState

__all__ = [
    "AverageMeter",
    "MixupCutmix",
    "ModelEma",
    "SoftTargetCrossEntropy",
    "TrainState",
    "Trainer",
    "accuracy",
    "build_optimizer",
    "build_scheduler",
]
