"""Tests for the MLP block and Transformer encoder."""

from __future__ import annotations

import pytest
import torch

from vit.models.encoder import EncoderBlock, TransformerEncoder
from vit.models.mlp import MLP


def test_mlp_shape_and_default_hidden() -> None:
    mlp = MLP(in_features=16)
    assert mlp.fc1.out_features == 16  # default hidden == in_features
    out = mlp(torch.randn(2, 5, 16))
    assert out.shape == (2, 5, 16)


def test_mlp_custom_widths() -> None:
    mlp = MLP(in_features=16, hidden_features=64, out_features=8, drop=0.1)
    assert mlp.fc1.out_features == 64
    assert mlp(torch.randn(3, 16)).shape == (3, 8)


def test_encoder_block_residual_shape() -> None:
    block = EncoderBlock(embed_dim=16, num_heads=2, mlp_ratio=2.0)
    x = torch.randn(2, 6, 16)
    out, weights = block(x)
    assert out.shape == x.shape
    assert weights is None


def test_encoder_block_returns_weights() -> None:
    block = EncoderBlock(embed_dim=16, num_heads=2)
    out, weights = block(torch.randn(1, 4, 16), need_weights=True)
    assert weights is not None and weights.shape == (1, 4, 4)


def test_encoder_stack_depth_and_drop_path_ramp() -> None:
    depth = 4
    enc = TransformerEncoder(depth=depth, embed_dim=16, num_heads=2, drop_path_rate=0.4)
    assert len(enc.blocks) == depth
    rates = [b.drop_path1.drop_prob for b in enc.blocks]
    # Linearly ramped from 0 to drop_path_rate.
    assert rates[0] == 0.0
    assert rates[-1] == pytest.approx(0.4)
    assert rates == sorted(rates)


def test_encoder_collects_attention_per_layer() -> None:
    enc = TransformerEncoder(depth=3, embed_dim=16, num_heads=2)
    x = torch.randn(2, 5, 16)
    out, attentions = enc(x, return_attention=True)
    assert out.shape == x.shape
    assert len(attentions) == 3
    assert all(a.shape == (2, 5, 5) for a in attentions)


def test_encoder_no_attention_by_default() -> None:
    enc = TransformerEncoder(depth=2, embed_dim=16, num_heads=2)
    _, attentions = enc(torch.randn(1, 5, 16))
    assert attentions == []
