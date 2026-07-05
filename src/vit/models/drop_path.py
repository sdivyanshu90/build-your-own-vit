"""Stochastic depth (DropPath).

Introduced by Huang et al. (2016, "Deep Networks with Stochastic Depth"), this
regularizer randomly drops entire residual branches during training. Each branch
that survives is rescaled by ``1/keep_prob`` so the expected value is unchanged
(inverted dropout). At inference it is a no-op. In ViTs the per-layer drop
probability is typically ramped linearly from ``0`` at the first block to
``drop_path_rate`` at the last.
"""

from __future__ import annotations

import torch
from torch import nn


def drop_path(
    x: torch.Tensor, drop_prob: float, training: bool, scale_by_keep: bool = True
) -> torch.Tensor:
    """Apply stochastic depth to a per-sample residual branch.

    Args:
        x: The residual-branch output, shape ``(B, ...)``.
        drop_prob: Probability of dropping the whole branch for a sample.
        training: If ``False`` or ``drop_prob == 0``, returns ``x`` unchanged.
        scale_by_keep: Rescale surviving samples by ``1/keep_prob``.
    """
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # Broadcast mask over all non-batch dims so an entire sample is kept/dropped.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Module wrapper around :func:`drop_path`."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}.")
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"
