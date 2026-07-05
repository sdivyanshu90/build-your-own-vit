"""Tests for the data module and transforms."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from vit.config import DataConfig
from vit.data import DataModule, build_eval_transforms, build_train_transforms


def test_fake_datamodule_shapes() -> None:
    cfg = DataConfig(dataset="fake", image_size=16, batch_size=8, num_workers=0)
    dm = DataModule(cfg).setup()
    assert dm.num_classes == 10
    assert len(dm.class_names) == 10
    images, targets = next(iter(dm.train_dataloader()))
    assert images.shape == (8, 3, 16, 16)
    assert targets.shape == (8,)


def test_datamodule_setup_idempotent() -> None:
    dm = DataModule(DataConfig(dataset="fake", image_size=16, num_workers=0))
    assert dm.setup() is dm.setup()


def test_val_and_test_loaders_present() -> None:
    dm = DataModule(DataConfig(dataset="fake", image_size=16, num_workers=0)).setup()
    assert dm.val_dataloader() is not None
    assert dm.test_dataloader() is not None


def test_transforms_output_tensor() -> None:
    cfg = DataConfig(dataset="fake", image_size=16, rand_augment=True)
    img = Image.new("RGB", (16, 16), (10, 20, 30))
    train_t = build_train_transforms(cfg)
    eval_t = build_eval_transforms(cfg)
    assert isinstance(train_t(img), torch.Tensor)
    out = eval_t(img)
    assert out.shape == (3, 16, 16)


def test_imagefolder_missing_dir_raises(tmp_path: Path) -> None:
    cfg = DataConfig(dataset="imagefolder", data_dir=str(tmp_path), image_size=16, num_workers=0)
    with pytest.raises(FileNotFoundError, match="train"):
        DataModule(cfg).setup()


def _make_imagefolder(root: Path, split: str, classes: int = 3, per_class: int = 4) -> None:
    for c in range(classes):
        d = root / split / f"class{c}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            Image.new("RGB", (16, 16), (c * 40, i * 20, 50)).save(d / f"{i}.png")


def test_imagefolder_with_explicit_val(tmp_path: Path) -> None:
    _make_imagefolder(tmp_path, "train")
    _make_imagefolder(tmp_path, "val", per_class=2)
    _make_imagefolder(tmp_path, "test", per_class=2)
    cfg = DataConfig(
        dataset="imagefolder",
        data_dir=str(tmp_path),
        image_size=16,
        batch_size=4,
        num_workers=0,
        val_split=0.0,
    )
    dm = DataModule(cfg).setup()
    assert dm.num_classes == 3
    assert dm.val_dataloader() is not None
    assert dm.test_dataloader() is not None
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape[1:] == (3, 16, 16)


def test_imagefolder_auto_split(tmp_path: Path) -> None:
    _make_imagefolder(tmp_path, "train", per_class=10)
    cfg = DataConfig(
        dataset="imagefolder",
        data_dir=str(tmp_path),
        image_size=16,
        batch_size=4,
        num_workers=0,
        val_split=0.2,
    )
    dm = DataModule(cfg).setup()
    assert dm.val_dataloader() is not None  # split off from train
    assert dm.test_dataloader() is None  # no test dir provided
