"""Image preprocessing and augmentation pipelines.

Two pipelines are derived from a :class:`~vit.config.DataConfig`:

* **Train** — geometric and photometric augmentation (random crop, horizontal
  flip, optional RandAugment) followed by tensor conversion and normalization.
* **Eval** — deterministic resize + normalization only.

The exact same normalization is applied at inference time by the
:class:`~vit.inference.predictor.Predictor`, so training and serving stay in
lockstep.
"""

from __future__ import annotations

from torchvision import transforms

from vit.config import DataConfig


def build_train_transforms(cfg: DataConfig) -> transforms.Compose:
    """Compose the training augmentation pipeline from a data config."""
    ops: list[object] = [
        transforms.RandomCrop(
            cfg.image_size,
            padding=cfg.random_crop_padding if cfg.random_crop_padding else None,
            padding_mode="reflect",
        )
    ]
    if cfg.horizontal_flip:
        ops.append(transforms.RandomHorizontalFlip())
    if cfg.rand_augment:
        ops.append(
            transforms.RandAugment(
                num_ops=cfg.rand_augment_num_ops,
                magnitude=cfg.rand_augment_magnitude,
            )
        )
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std),
    ]
    return transforms.Compose(ops)


def build_eval_transforms(cfg: DataConfig) -> transforms.Compose:
    """Compose the deterministic evaluation/inference pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )
