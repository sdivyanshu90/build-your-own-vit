"""Mixup and CutMix batch augmentation with soft targets.

Mixup (Zhang et al., 2018) and CutMix (Yun et al., 2019) are strong regularizers
for ViTs. Both interpolate a *pair* of samples and their labels:

* **Mixup** blends whole images: ``x = λ·x_a + (1-λ)·x_b``.
* **CutMix** pastes a rectangular patch of one image onto another; ``λ`` is the
  remaining area fraction.

Both produce soft (probabilistic) targets, so they are paired with
:class:`SoftTargetCrossEntropy`. Label smoothing is folded into the soft target
construction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _one_hot(
    targets: torch.Tensor, num_classes: int, on_value: float, off_value: float
) -> torch.Tensor:
    """Build a smoothed one-hot target matrix ``(B, num_classes)``."""
    out = torch.full(
        (targets.shape[0], num_classes),
        off_value,
        device=targets.device,
        dtype=torch.float32,
    )
    return out.scatter_(1, targets.long().view(-1, 1), on_value)


def _rand_bbox(
    height: int, width: int, lam: float, rng: np.random.Generator
) -> tuple[int, int, int, int]:
    """Sample a CutMix bounding box whose area ~ ``(1-lam)`` of the image."""
    ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    cy, cx = rng.integers(height), rng.integers(width)
    y1 = int(np.clip(cy - cut_h // 2, 0, height))
    y2 = int(np.clip(cy + cut_h // 2, 0, height))
    x1 = int(np.clip(cx - cut_w // 2, 0, width))
    x2 = int(np.clip(cx + cut_w // 2, 0, width))
    return y1, y2, x1, x2


class MixupCutmix:
    """Applies Mixup or CutMix (chosen per-batch) and returns soft targets.

    Args:
        num_classes: Number of target classes.
        mixup_alpha: Beta distribution alpha for Mixup (0 disables Mixup).
        cutmix_alpha: Beta distribution alpha for CutMix (0 disables CutMix).
        prob: Probability of applying any mixing to a given batch.
        switch_prob: When both are enabled, probability of choosing CutMix.
        label_smoothing: Smoothing applied to the constructed soft targets.
        seed: Seed for the internal NumPy RNG (reproducible mixing).
    """

    def __init__(
        self,
        num_classes: int,
        *,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        label_smoothing: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self._rng = np.random.default_rng(seed)

    @property
    def enabled(self) -> bool:
        return self.mixup_alpha > 0 or self.cutmix_alpha > 0

    def _soft_target(self, target: torch.Tensor, lam: float) -> torch.Tensor:
        off = self.label_smoothing / self.num_classes
        on = 1.0 - self.label_smoothing + off
        y1 = _one_hot(target, self.num_classes, on, off)
        y2 = y1.flip(0)  # pair each sample with its mirror in the batch
        return lam * y1 + (1.0 - lam) * y2

    def __call__(self, x: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mixed_input, soft_target)``.

        If mixing is disabled or skipped this batch, a smoothed one-hot target
        is still returned so the training loop can always use soft-target loss.
        """
        if not self.enabled or self._rng.random() > self.prob:
            return x, self._soft_target(target, lam=1.0)

        use_cutmix = self.cutmix_alpha > 0 and (
            self.mixup_alpha <= 0 or self._rng.random() < self.switch_prob
        )
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        lam = float(self._rng.beta(alpha, alpha))
        flipped = x.flip(0)

        if use_cutmix:
            _, _, height, width = x.shape
            y1, y2, x1, x2 = _rand_bbox(height, width, lam, self._rng)
            x = x.clone()
            x[:, :, y1:y2, x1:x2] = flipped[:, :, y1:y2, x1:x2]
            # Adjust lambda to the true pixel ratio actually replaced.
            area = (y2 - y1) * (x2 - x1)
            lam = 1.0 - area / (height * width)
        else:
            x = lam * x + (1.0 - lam) * flipped

        return x, self._soft_target(target, lam)


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy against soft (probabilistic) targets.

    Equivalent to ``-Σ target · log_softmax(logits)`` averaged over the batch.
    Used together with :class:`MixupCutmix`.
    """

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1)
        return loss.mean()
