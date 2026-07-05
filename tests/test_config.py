"""Tests for the configuration schema and YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vit.config import (
    Config,
    DataConfig,
    ModelConfig,
    load_config,
    save_config,
)


def test_preset_fills_defaults() -> None:
    cfg = ModelConfig.from_preset("vit_small", num_classes=10, image_size=32, patch_size=4)
    assert cfg.embed_dim == 384
    assert cfg.depth == 12
    assert cfg.num_heads == 6


def test_explicit_override_beats_preset() -> None:
    cfg = ModelConfig(
        preset="vit_small", embed_dim=128, num_heads=4, image_size=32, patch_size=4, num_classes=10
    )
    assert cfg.embed_dim == 128  # explicit wins
    assert cfg.depth == 12  # preset fills the rest


def test_unknown_preset_raises() -> None:
    with pytest.raises(ValidationError, match="Unknown preset"):
        ModelConfig.from_preset("vit_giant", num_classes=1, image_size=8, patch_size=4)


def test_patch_must_divide_image() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        ModelConfig(image_size=30, patch_size=4)


def test_embed_dim_divisible_by_heads() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        ModelConfig(embed_dim=100, num_heads=3, image_size=16, patch_size=4)


def test_dropout_upper_bound() -> None:
    with pytest.raises(ValidationError, match="< 1.0"):
        ModelConfig(drop_rate=1.5, image_size=16, patch_size=4)


def test_num_patches_and_seq_len() -> None:
    cfg = ModelConfig(image_size=32, patch_size=8, pool="cls")
    assert cfg.num_patches == 16
    assert cfg.seq_len == 17
    assert cfg.model_copy(update={"pool": "mean"}).seq_len == 16


def test_extra_keys_forbidden() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(image_size=16, patch_size=4, bogus=123)


def test_cross_section_image_size_mismatch() -> None:
    with pytest.raises(ValidationError, match="image_size"):
        Config(
            model=ModelConfig(image_size=32, patch_size=4, num_classes=10),
            data=DataConfig(dataset="cifar10", image_size=16),
        )


def test_cross_section_num_classes_mismatch() -> None:
    with pytest.raises(ValidationError, match="num_classes"):
        Config(
            model=ModelConfig(image_size=32, patch_size=4, num_classes=42),
            data=DataConfig(dataset="cifar10", image_size=32),
        )


def test_mean_std_wrong_length() -> None:
    with pytest.raises(ValidationError, match="3 channel"):
        DataConfig(mean=(0.1, 0.2))


def test_load_config_with_overrides(tmp_path: Path) -> None:
    payload = {
        "model": {
            "image_size": 16,
            "patch_size": 4,
            "num_classes": 10,
            "embed_dim": 16,
            "depth": 1,
            "num_heads": 2,
        },
        "data": {"dataset": "fake", "image_size": 16},
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(payload))
    cfg = load_config(path, **{"train.epochs": 3, "optim.lr": 0.5})
    assert cfg.train.epochs == 3
    assert cfg.optim.lr == 0.5


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_load_config_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_save_and_roundtrip(tmp_path: Path, tiny_config: Config) -> None:
    path = save_config(tiny_config, tmp_path / "out" / "cfg.yaml")
    assert path.is_file()
    reloaded = load_config(path)
    assert reloaded.model.embed_dim == tiny_config.model.embed_dim
    assert reloaded.data.dataset == tiny_config.data.dataset


def test_to_yaml_is_parseable(tiny_config: Config) -> None:
    parsed = yaml.safe_load(tiny_config.to_yaml())
    assert parsed["run_name"] == "test-run"
