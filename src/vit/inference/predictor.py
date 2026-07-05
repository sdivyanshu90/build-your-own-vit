"""High-level inference API.

:class:`Predictor` loads a checkpoint, rebuilds the exact model architecture from
the config stored inside it, restores the preprocessing pipeline, and exposes a
simple ``predict`` method over PIL images. Because the architecture and
normalization are read from the checkpoint, a caller never has to know the
training hyper-parameters — the file is fully self-describing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from vit.config import ModelConfig
from vit.models.factory import build_model
from vit.utils.checkpoint import load_checkpoint
from vit.utils.device import resolve_device
from vit.utils.logging import get_logger

logger = get_logger(__name__)

# Fallback normalization if a checkpoint predates the ``preprocess`` field.
_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
_DEFAULT_STD = (0.2470, 0.2435, 0.2616)


@dataclass(slots=True, frozen=True)
class Prediction:
    """A single ranked class prediction."""

    label: str
    index: int
    probability: float


class Predictor:
    """Loads a trained ViT and classifies images.

    Prefer :meth:`from_checkpoint` for construction.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: list[str],
        transform: transforms.Compose,
        device: torch.device,
        image_size: int,
    ) -> None:
        self.model = model.to(device).eval()
        self.class_names = class_names
        self.transform = transform
        self.device = device
        self.image_size = image_size

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        use_ema: bool = True,
    ) -> Predictor:
        """Build a predictor from a checkpoint file.

        Args:
            path: Path to a ``.pt`` checkpoint.
            device: Device preference (``auto``/``cpu``/``cuda``/...).
            use_ema: Prefer EMA weights if present in the checkpoint.
        """
        resolved = resolve_device(device)
        ckpt = load_checkpoint(path, map_location=resolved)
        model_cfg = ModelConfig.model_validate(ckpt.model_config)
        model = build_model(model_cfg)

        state = ckpt.model_state
        if use_ema and ckpt.ema_state:
            state = ckpt.ema_state
            logger.info("Loading EMA weights for inference")
        model.load_state_dict(state)

        class_names = ckpt.class_names or [str(i) for i in range(model_cfg.num_classes)]
        transform = cls._build_transform(ckpt.preprocess, model_cfg.image_size)
        return cls(model, class_names, transform, resolved, model_cfg.image_size)

    @staticmethod
    def _build_transform(
        preprocess: dict[str, Any] | None, image_size: int
    ) -> transforms.Compose:
        pp: dict[str, Any] = preprocess or {}
        size = int(pp.get("image_size", image_size))
        mean = tuple(pp.get("mean", _DEFAULT_MEAN))
        std = tuple(pp.get("std", _DEFAULT_STD))
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    # -- prediction --------------------------------------------------------
    def _to_batch(self, images: Image.Image | list[Image.Image]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]
        if not images:
            raise ValueError("No images provided for prediction.")
        tensors = [self.transform(img.convert("RGB")) for img in images]
        return torch.stack(tensors).to(self.device)

    @torch.no_grad()
    def predict_proba(self, images: Image.Image | list[Image.Image]) -> torch.Tensor:
        """Return class probabilities of shape ``(B, num_classes)``."""
        batch = self._to_batch(images)
        logits = self.model(batch)
        return F.softmax(logits, dim=-1)

    def predict(
        self, images: Image.Image | list[Image.Image], *, top_k: int = 5
    ) -> list[list[Prediction]]:
        """Classify one or more images.

        Args:
            images: A single PIL image or a list of them.
            top_k: Number of ranked predictions to return per image.

        Returns:
            For each input image, a list of :class:`Prediction` sorted by
            descending probability.
        """
        probs = self.predict_proba(images)
        k = min(top_k, probs.shape[1])
        top_probs, top_idx = probs.topk(k, dim=-1)
        results: list[list[Prediction]] = []
        for row_probs, row_idx in zip(top_probs.tolist(), top_idx.tolist(), strict=True):
            results.append(
                [
                    Prediction(label=self.class_names[i], index=int(i), probability=float(p))
                    for p, i in zip(row_probs, row_idx, strict=True)
                ]
            )
        return results
