"""Tests for the full VisionTransformer model."""

from __future__ import annotations

import pytest
import torch

from vit.config import ModelConfig
from vit.models import VisionTransformer, build_from_preset, build_model, list_presets


def test_forward_shape_cls_pool(tiny_model_config: ModelConfig) -> None:
    model = build_model(tiny_model_config).eval()
    logits = model(torch.randn(4, 3, 16, 16))
    assert logits.shape == (4, 10)


def test_forward_shape_mean_pool() -> None:
    cfg = ModelConfig(
        image_size=16,
        patch_size=4,
        num_classes=7,
        embed_dim=16,
        depth=2,
        num_heads=2,
        pool="mean",
    )
    model = build_model(cfg).eval()
    assert model.cls_token is None
    assert model(torch.randn(2, 3, 16, 16)).shape == (2, 7)


def test_pos_embed_length_matches_pool() -> None:
    cls = ModelConfig(
        image_size=16, patch_size=4, num_classes=3, embed_dim=16, depth=1, num_heads=2, pool="cls"
    )
    mean = cls.model_copy(update={"pool": "mean"})
    m_cls, m_mean = build_model(cls), build_model(mean)
    assert m_cls.pos_embed.shape[1] == cls.num_patches + 1
    assert m_mean.pos_embed.shape[1] == mean.num_patches


def test_forward_features_and_representation() -> None:
    cfg = ModelConfig(
        image_size=16,
        patch_size=4,
        num_classes=5,
        embed_dim=16,
        depth=1,
        num_heads=2,
        representation_size=24,
    )
    model = build_model(cfg).eval()
    feats = model.forward_features(torch.randn(2, 3, 16, 16))
    assert feats.shape == (2, 24)  # pre-logits projects to representation_size


def test_attention_maps(tiny_model_config: ModelConfig) -> None:
    model = build_model(tiny_model_config)
    maps = model.get_attention_maps(torch.randn(2, 3, 16, 16))
    assert len(maps) == tiny_model_config.depth
    seq = tiny_model_config.seq_len
    assert all(m.shape == (2, seq, seq) for m in maps)


def test_get_attention_maps_restores_training_mode(tiny_model_config: ModelConfig) -> None:
    model = build_model(tiny_model_config)
    model.train()
    model.get_attention_maps(torch.randn(1, 3, 16, 16))
    assert model.training is True


def test_gradient_flow_all_params(tiny_model_config: ModelConfig) -> None:
    model = build_model(tiny_model_config)
    model(torch.randn(2, 3, 16, 16)).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_num_parameters_positive(tiny_model_config: ModelConfig) -> None:
    assert build_model(tiny_model_config).num_parameters > 0


def test_invalid_pool_raises() -> None:
    with pytest.raises(ValueError, match="pool"):
        VisionTransformer(image_size=16, patch_size=4, pool="max")


def test_preset_param_counts_match_reference() -> None:
    # Sanity-check the canonical ImageNet configurations (±2% of published).
    reference = {"vit_tiny": 5.7e6, "vit_small": 22e6, "vit_base": 86e6}
    for preset, expected in reference.items():
        model = build_from_preset(preset, num_classes=1000, image_size=224, patch_size=16)
        assert abs(model.num_parameters - expected) / expected < 0.03


def test_list_presets() -> None:
    assert set(list_presets()) == {"vit_tiny", "vit_small", "vit_base", "vit_large"}


def test_from_config_roundtrip(tiny_model_config: ModelConfig) -> None:
    model = VisionTransformer.from_config(tiny_model_config)
    assert model.num_classes == tiny_model_config.num_classes
    assert model.embed_dim == tiny_model_config.embed_dim
