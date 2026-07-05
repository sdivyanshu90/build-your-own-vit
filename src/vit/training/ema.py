"""Exponential Moving Average (EMA) of model weights.

Maintaining an EMA of the weights (a.k.a. "model averaging" or Polyak
averaging) typically yields a smoother, better-generalizing model for
evaluation and serving. The EMA is updated after every optimizer step:

    ema = decay · ema + (1 - decay) · weights

A near-1 decay (e.g. 0.9999) makes the EMA a long-horizon average.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator

import torch
from torch import nn


class ModelEma:
    """Tracks an EMA copy of a model's parameters and buffers.

    Args:
        model: The model whose weights are averaged.
        decay: EMA decay factor in ``(0, 1)``; higher = slower to adapt.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}.")
        self.decay = decay
        # A frozen deep copy on the same device; never receives gradients.
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update the EMA weights from ``model``'s current parameters/buffers."""
        for ema_p, model_p in zip(
            self._state_iter(self.module), self._state_iter(model), strict=True
        ):
            if ema_p.is_floating_point():
                ema_p.mul_(self.decay).add_(model_p.detach(), alpha=1.0 - self.decay)
            else:
                # Integer buffers (e.g. num_batches_tracked) are copied verbatim.
                ema_p.copy_(model_p)

    @staticmethod
    def _state_iter(module: nn.Module) -> Iterator[torch.Tensor]:
        yield from module.parameters()
        yield from module.buffers()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.module.load_state_dict(state)
