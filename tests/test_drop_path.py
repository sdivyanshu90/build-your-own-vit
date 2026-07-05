"""Tests for stochastic depth (DropPath)."""

from __future__ import annotations

import pytest
import torch

from vit.models.drop_path import DropPath, drop_path


def test_identity_in_eval() -> None:
    module = DropPath(0.5).eval()
    x = torch.randn(4, 8)
    torch.testing.assert_close(module(x), x)


def test_identity_when_prob_zero() -> None:
    x = torch.randn(4, 8)
    torch.testing.assert_close(drop_path(x, 0.0, training=True), x)


def test_expected_value_preserved_on_average() -> None:
    torch.manual_seed(0)
    x = torch.ones(20000, 1)
    out = drop_path(x, 0.3, training=True)
    # Inverted dropout keeps E[out] ≈ E[x] = 1.
    assert out.mean().item() == pytest.approx(1.0, abs=0.05)


def test_drops_whole_samples() -> None:
    torch.manual_seed(1)
    x = torch.ones(100, 3, 4)
    out = drop_path(x, 0.5, training=True)
    # Each sample is entirely kept (scaled) or entirely zero.
    per_sample_unique = [len(torch.unique(out[i])) for i in range(out.shape[0])]
    assert all(u == 1 for u in per_sample_unique)


def test_rejects_out_of_range_prob() -> None:
    with pytest.raises(ValueError):
        DropPath(1.0)
    with pytest.raises(ValueError):
        DropPath(-0.1)


def test_extra_repr() -> None:
    assert "drop_prob" in DropPath(0.25).extra_repr()
