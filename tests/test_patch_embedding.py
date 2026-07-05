"""Tests for the patch embedding layer."""

from __future__ import annotations

import pytest
import torch

from vit.models.patch_embedding import PatchEmbedding


def test_output_shape_and_num_patches() -> None:
    layer = PatchEmbedding(image_size=32, patch_size=4, in_channels=3, embed_dim=48)
    assert layer.num_patches == 64
    assert layer.grid_size == 8
    out = layer(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 64, 48)


def test_rejects_indivisible_patch_size() -> None:
    with pytest.raises(ValueError, match="divisible"):
        PatchEmbedding(image_size=30, patch_size=4, in_channels=3, embed_dim=8)


def test_rejects_wrong_input_rank() -> None:
    layer = PatchEmbedding(16, 4, 3, 8)
    with pytest.raises(ValueError, match="4-D"):
        layer(torch.randn(3, 16, 16))


def test_rejects_wrong_spatial_size() -> None:
    layer = PatchEmbedding(16, 4, 3, 8)
    with pytest.raises(ValueError, match="does not match"):
        layer(torch.randn(1, 3, 24, 24))


def test_grayscale_channels() -> None:
    layer = PatchEmbedding(8, 2, in_channels=1, embed_dim=6)
    out = layer(torch.randn(1, 1, 8, 8))
    assert out.shape == (1, 16, 6)
