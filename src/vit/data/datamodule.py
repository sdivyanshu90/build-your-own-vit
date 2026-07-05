"""Dataset assembly and DataLoader construction.

The :class:`DataModule` centralizes dataset selection, the train/val split, and
DataLoader wiring so the training loop stays dataset-agnostic. Supported
backends:

* ``cifar10`` / ``cifar100`` — torchvision datasets (auto-downloaded).
* ``imagefolder`` — a standard ``root/split/class/*.jpg`` directory tree.
* ``fake`` — synthetic in-memory data for tests and smoke runs (no download).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from vit.config import DataConfig
from vit.data.transforms import build_eval_transforms, build_train_transforms
from vit.utils.logging import get_logger
from vit.utils.seed import worker_init_fn

logger = get_logger(__name__)


class DataModule:
    """Builds train/val/test :class:`~torch.utils.data.DataLoader` objects.

    Call :meth:`setup` once before requesting loaders.
    """

    def __init__(self, cfg: DataConfig, *, seed: int = 42) -> None:
        self.cfg = cfg
        self.seed = seed
        self.train_transform = build_train_transforms(cfg)
        self.eval_transform = build_eval_transforms(cfg)
        self._train: Dataset | None = None
        self._val: Dataset | None = None
        self._test: Dataset | None = None
        self._class_names: list[str] = []

    # -- setup -------------------------------------------------------------
    def setup(self) -> DataModule:
        """Instantiate datasets. Idempotent; returns ``self`` for chaining."""
        if self._train is not None:
            return self
        if self.cfg.dataset in {"cifar10", "cifar100"}:
            self._setup_cifar()
        elif self.cfg.dataset == "imagefolder":
            self._setup_imagefolder()
        elif self.cfg.dataset == "fake":
            self._setup_fake()
        else:  # pragma: no cover - guarded by config Literal
            raise ValueError(f"Unsupported dataset {self.cfg.dataset!r}.")
        logger.info(
            "DataModule ready",
            extra={
                "dataset": self.cfg.dataset,
                "num_classes": self.num_classes,
                "train_size": len(self._train),  # type: ignore[arg-type]
                "val_size": len(self._val) if self._val else 0,  # type: ignore[arg-type]
                "test_size": len(self._test) if self._test else 0,  # type: ignore[arg-type]
            },
        )
        return self

    def _setup_cifar(self) -> None:
        ctor = datasets.CIFAR10 if self.cfg.dataset == "cifar10" else datasets.CIFAR100
        train_full = ctor(
            self.cfg.data_dir,
            train=True,
            download=self.cfg.download,
            transform=self.train_transform,
        )
        val_full = ctor(
            self.cfg.data_dir,
            train=True,
            download=False,
            transform=self.eval_transform,
        )
        self._test = ctor(
            self.cfg.data_dir,
            train=False,
            download=self.cfg.download,
            transform=self.eval_transform,
        )
        self._class_names = list(train_full.classes)
        self._split_train_val(train_full, val_full)

    def _setup_imagefolder(self) -> None:
        root = Path(self.cfg.data_dir)
        train_dir = root / "train"
        if not train_dir.is_dir():
            raise FileNotFoundError(
                f"ImageFolder expects a 'train' subdir under {root}."
            )
        train_full = datasets.ImageFolder(train_dir, transform=self.train_transform)
        self._class_names = list(train_full.classes)

        val_dir = root / "val"
        test_dir = root / "test"
        if val_dir.is_dir():
            self._train = train_full
            self._val = datasets.ImageFolder(val_dir, transform=self.eval_transform)
        else:
            val_full = datasets.ImageFolder(train_dir, transform=self.eval_transform)
            self._split_train_val(train_full, val_full)
        if test_dir.is_dir():
            self._test = datasets.ImageFolder(test_dir, transform=self.eval_transform)

    def _setup_fake(self) -> None:
        # 10-class synthetic RGB dataset for tests/smoke runs; no network needed.
        num_classes = 10
        shape = (3, self.cfg.image_size, self.cfg.image_size)
        self._train = datasets.FakeData(
            size=256,
            image_size=shape,
            num_classes=num_classes,
            transform=self.train_transform,
        )
        self._val = datasets.FakeData(
            size=64,
            image_size=shape,
            num_classes=num_classes,
            transform=self.eval_transform,
        )
        self._test = datasets.FakeData(
            size=64,
            image_size=shape,
            num_classes=num_classes,
            transform=self.eval_transform,
        )
        self._class_names = [f"class_{i}" for i in range(num_classes)]

    def _split_train_val(self, train_full: Dataset, val_full: Dataset) -> None:
        """Partition a single training set into disjoint train/val subsets."""
        n = len(train_full)  # type: ignore[arg-type]
        n_val = int(round(n * self.cfg.val_split))
        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=generator).tolist()
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        self._train = Subset(train_full, train_idx)
        self._val = Subset(val_full, val_idx) if n_val > 0 else None

    # -- properties --------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    # -- loaders -----------------------------------------------------------
    def _loader(self, dataset: Dataset | None, *, shuffle: bool) -> DataLoader | None:
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=shuffle,  # drop last only for training for stable batch stats
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        loader = self._loader(self._train, shuffle=True)
        if loader is None:  # pragma: no cover - setup guarantees a train set
            raise RuntimeError("Call setup() before requesting the train loader.")
        return loader

    def val_dataloader(self) -> DataLoader | None:
        return self._loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader | None:
        return self._loader(self._test, shuffle=False)
