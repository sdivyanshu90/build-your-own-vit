"""Patch embedding: image → sequence of patch tokens.

A Vision Transformer treats an image as a sequence. The image is split into a
grid of non-overlapping ``patch_size × patch_size`` patches, and each patch is
linearly projected to ``embed_dim``. This is implemented efficiently as a single
strided convolution whose kernel and stride both equal the patch size — the
convolution *is* the per-patch linear projection.

Input  : ``(B, C, H, W)``
Output : ``(B, N, D)`` where ``N = (H/P)·(W/P)`` and ``D = embed_dim``.
"""

from __future__ import annotations

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Convert an image into a sequence of linearly-projected patch embeddings.

    Args:
        image_size: Side length of the (square) input image.
        patch_size: Side length of each square patch. Must divide ``image_size``.
        in_channels: Number of input channels (3 for RGB).
        embed_dim: Dimensionality of each patch token.

    Raises:
        ValueError: If ``image_size`` is not divisible by ``patch_size``.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})."
            )
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        # Kernel == stride == patch_size makes each output pixel one patch token.
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` of shape ``(B, C, H, W)`` to tokens ``(B, N, D)``."""
        if x.ndim != 4:
            raise ValueError(f"Expected a 4-D tensor (B,C,H,W), got shape {tuple(x.shape)}.")
        _, _, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Input size ({height}x{width}) does not match the configured "
                f"image_size ({self.image_size}x{self.image_size})."
            )
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
