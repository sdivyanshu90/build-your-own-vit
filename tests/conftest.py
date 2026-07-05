"""Shared pytest fixtures.

All fixtures run on CPU with the synthetic ``fake`` dataset so the suite is
hermetic (no downloads, no GPU) and fast. A single small model is trained once
per session and reused by the inference/serving/export tests.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from PIL import Image

from vit.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimConfig,
    SchedulerConfig,
    TrainConfig,
)

# Keep CPU tests snappy and deterministic.
torch.set_num_threads(2)


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    """A deliberately small ViT config for fast unit tests."""
    return ModelConfig(
        image_size=16,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )


@pytest.fixture
def tiny_config(tmp_path: Path) -> Config:
    """A full training Config on synthetic data, CPU, 1 epoch."""
    return Config(
        run_name="test-run",
        model=ModelConfig(
            image_size=16,
            patch_size=4,
            num_classes=10,
            embed_dim=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
        ),
        data=DataConfig(
            dataset="fake",
            image_size=16,
            batch_size=8,
            num_workers=0,
            download=False,
        ),
        optim=OptimConfig(lr=1e-3),
        scheduler=SchedulerConfig(warmup_epochs=0),
        train=TrainConfig(
            epochs=1,
            precision="fp32",
            device="cpu",
            output_dir=str(tmp_path / "outputs"),
            log_interval=1,
        ),
    )


@pytest.fixture(scope="session")
def trained_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a tiny model once and return the path to ``last.pt``."""
    from vit.data import DataModule
    from vit.training import Trainer

    out = tmp_path_factory.mktemp("trained")
    config = Config(
        run_name="ckpt",
        model=ModelConfig(
            image_size=16,
            patch_size=4,
            num_classes=10,
            embed_dim=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
        ),
        data=DataConfig(dataset="fake", image_size=16, batch_size=8, num_workers=0),
        scheduler=SchedulerConfig(warmup_epochs=0),
        train=TrainConfig(
            epochs=1,
            precision="fp32",
            device="cpu",
            output_dir=str(out),
            use_ema=True,
        ),
    )
    trainer = Trainer(config, DataModule(config.data), progress=False)
    trainer.fit(max_steps=3)
    return trainer.output_dir / "last.pt"


@pytest.fixture
def rgb_image() -> Image.Image:
    """A small deterministic RGB image."""
    return Image.new("RGB", (16, 16), (123, 45, 200))


@pytest.fixture
def png_bytes(rgb_image: Image.Image) -> bytes:
    """PNG-encoded bytes of the sample image."""
    buf = io.BytesIO()
    rgb_image.save(buf, format="PNG")
    return buf.getvalue()
