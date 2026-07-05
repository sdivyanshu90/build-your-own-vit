"""Build Your Own ViT — a production-grade Vision Transformer in PyTorch.

The public API is intentionally small and stable. Heavy submodules (training,
serving) are imported lazily to keep ``import vit`` cheap and side-effect free,
which matters for CLI start-up latency and for serverless cold starts.

Example:
    >>> from vit import build_model, ModelConfig
    >>> model = build_model(ModelConfig(preset="vit_tiny", num_classes=10,
    ...                                 image_size=32, patch_size=4))
    >>> int(sum(p.numel() for p in model.parameters()) > 0)
    1
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "1.0.0"

__all__ = [
    "ModelConfig",
    "VisionTransformer",
    "__version__",
    "build_model",
    "list_presets",
]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vit.config import ModelConfig
    from vit.models import VisionTransformer, build_model, list_presets


def __getattr__(name: str) -> object:
    """Lazily resolve top-level attributes to avoid import-time heavy work."""
    if name in {"VisionTransformer", "build_model", "list_presets"}:
        from vit import models

        return getattr(models, name)
    if name == "ModelConfig":
        from vit.config import ModelConfig

        return ModelConfig
    raise AttributeError(f"module 'vit' has no attribute {name!r}")
