"""Learning-rate schedules.

Vision Transformers are sensitive to the LR schedule. The standard recipe is a
short linear **warmup** (avoiding the early instability of large LRs on a
freshly-initialized model) followed by **cosine decay** to a small floor. The
schedule here operates per optimizer *step* (not per epoch) for smooth updates.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from vit.config import SchedulerConfig


def build_scheduler(
    optimizer: Optimizer,
    cfg: SchedulerConfig,
    *,
    steps_per_epoch: int,
    epochs: int,
    base_lr: float,
) -> LambdaLR:
    """Build a per-step warmup+cosine (or constant) LR scheduler.

    The returned scheduler multiplies each param group's base LR by a factor in
    ``[0, 1]``; ``base_lr`` is used to compute the warmup start / min-lr floors
    as fractions of the peak.

    Args:
        optimizer: The optimizer to schedule.
        cfg: Scheduler configuration.
        steps_per_epoch: Optimizer steps in one epoch (after grad accumulation).
        epochs: Total number of epochs.
        base_lr: Peak learning rate (used to derive relative floors).
    """
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    warmup_start = cfg.warmup_start_lr / base_lr if base_lr > 0 else 0.0
    min_factor = cfg.min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if cfg.name == "none":
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            # Linear ramp from warmup_start factor up to 1.0.
            progress = step / warmup_steps
            return warmup_start + (1.0 - warmup_start) * progress
        # Cosine decay from 1.0 down to min_factor over the remaining steps.
        denom = max(total_steps - warmup_steps, 1)
        progress = min((step - warmup_steps) / denom, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return LambdaLR(optimizer, lr_lambda)
