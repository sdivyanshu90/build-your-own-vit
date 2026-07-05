"""Position-wise feed-forward network (the Transformer MLP block).

Applied identically to every token: two linear layers with a GELU
non-linearity and dropout in between. The hidden width is
``embed_dim × mlp_ratio`` (typically 4×), which is where most of a ViT's
parameters live.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class MLP(nn.Module):
    """Two-layer MLP with GELU activation used inside each encoder block.

    Args:
        in_features: Input/output width (``embed_dim``).
        hidden_features: Hidden width. Defaults to ``in_features``.
        out_features: Output width. Defaults to ``in_features``.
        activation: Activation module factory. Defaults to :class:`~torch.nn.GELU`.
        drop: Dropout probability applied after each linear layer.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        *,
        activation: Callable[[], nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x
