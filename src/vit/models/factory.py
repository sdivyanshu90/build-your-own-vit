"""Model construction helpers and preset registry.

:func:`build_model` is the single entry point for turning a validated
:class:`~vit.config.ModelConfig` into a :class:`~vit.models.vit.VisionTransformer`.
Building always goes through the config so that presets, validation, and
checkpoint round-tripping stay consistent.
"""

from __future__ import annotations

from vit.config import PRESETS, ModelConfig
from vit.models.vit import VisionTransformer


def build_model(config: ModelConfig) -> VisionTransformer:
    """Construct a :class:`VisionTransformer` from a :class:`ModelConfig`."""
    return VisionTransformer.from_config(config)


def build_from_preset(
    preset: str,
    *,
    num_classes: int,
    image_size: int,
    patch_size: int,
    **overrides: object,
) -> VisionTransformer:
    """Convenience constructor from a preset name plus required task geometry."""
    config = ModelConfig.from_preset(
        preset,
        num_classes=num_classes,
        image_size=image_size,
        patch_size=patch_size,
        **overrides,
    )
    return build_model(config)


def list_presets() -> list[str]:
    """Return the sorted names of available architecture presets."""
    return sorted(PRESETS)
