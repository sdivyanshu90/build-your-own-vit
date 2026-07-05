"""Tests for the top-level package surface and lazy imports."""

from __future__ import annotations

import pytest

import vit


def test_version_exposed() -> None:
    assert isinstance(vit.__version__, str)
    assert vit.__version__.count(".") >= 2


def test_lazy_model_config() -> None:
    from vit.config import ModelConfig

    assert vit.ModelConfig is ModelConfig


def test_lazy_model_symbols() -> None:
    assert vit.build_model.__name__ == "build_model"
    assert callable(vit.VisionTransformer)
    assert "vit_tiny" in vit.list_presets()


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        _ = vit.does_not_exist
