"""Tests for multi-head self-attention."""

from __future__ import annotations

import pytest
import torch

from vit.models.attention import MultiHeadSelfAttention


def test_output_shape() -> None:
    attn = MultiHeadSelfAttention(embed_dim=32, num_heads=4)
    x = torch.randn(2, 10, 32)
    out, weights = attn(x)
    assert out.shape == (2, 10, 32)
    assert weights is None


def test_returns_attention_weights_when_requested() -> None:
    attn = MultiHeadSelfAttention(embed_dim=16, num_heads=2)
    x = torch.randn(3, 5, 16)
    out, weights = attn(x, need_weights=True)
    assert out.shape == (3, 5, 16)
    assert weights is not None
    assert weights.shape == (3, 5, 5)
    # Attention rows are probability distributions → sum to 1.
    torch.testing.assert_close(weights.sum(-1), torch.ones(3, 5), rtol=1e-4, atol=1e-4)


def test_fused_and_manual_agree_in_eval() -> None:
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(embed_dim=16, num_heads=2, attn_drop=0.0).eval()
    x = torch.randn(2, 7, 16)
    fused, _ = attn(x, need_weights=False)
    manual, _ = attn(x, need_weights=True)
    torch.testing.assert_close(fused, manual, rtol=1e-4, atol=1e-4)


def test_rejects_bad_embed_head_divisibility() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadSelfAttention(embed_dim=30, num_heads=4)


def test_rejects_wrong_input_width() -> None:
    attn = MultiHeadSelfAttention(embed_dim=32, num_heads=4)
    with pytest.raises(ValueError, match="Expected input"):
        attn(torch.randn(2, 10, 16))


def test_gradients_flow() -> None:
    attn = MultiHeadSelfAttention(embed_dim=16, num_heads=2)
    x = torch.randn(1, 4, 16, requires_grad=True)
    out, _ = attn(x)
    out.sum().backward()
    assert x.grad is not None
    assert attn.qkv.weight.grad is not None
