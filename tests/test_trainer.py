"""Tests for the training engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from vit.config import Config, DataConfig, ModelConfig, SchedulerConfig, TrainConfig
from vit.data import DataModule
from vit.training import Trainer


def _config(tmp_path: Path, **train_overrides: object) -> Config:
    train_kwargs: dict[str, object] = {
        "epochs": 1,
        "precision": "fp32",
        "device": "cpu",
        "output_dir": str(tmp_path / "out"),
        "use_ema": False,
    }
    train_kwargs.update(train_overrides)
    return Config(
        model=ModelConfig(
            image_size=16,
            patch_size=4,
            num_classes=10,
            embed_dim=16,
            depth=1,
            num_heads=2,
            mlp_ratio=2.0,
        ),
        data=DataConfig(dataset="fake", image_size=16, batch_size=8, num_workers=0),
        scheduler=SchedulerConfig(warmup_epochs=0),
        train=TrainConfig(**train_kwargs),  # type: ignore[arg-type]
    )


def test_fit_produces_history_and_checkpoints(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    trainer = Trainer(cfg, DataModule(cfg.data), progress=False)
    state = trainer.fit(max_steps=3)
    assert state.history
    assert "val_acc1" in state.history[-1]
    assert (trainer.output_dir / "last.pt").is_file()
    assert (trainer.output_dir / "best.pt").is_file()


def test_evaluate_returns_metrics(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    trainer = Trainer(cfg, DataModule(cfg.data), progress=False)
    metrics = trainer.evaluate(trainer.datamodule.val_dataloader(), max_batches=1)
    assert set(metrics) == {"val_loss", "val_acc1", "val_acc5"}
    assert 0.0 <= metrics["val_acc1"] <= 100.0


def test_ema_and_mixup_paths(tmp_path: Path) -> None:
    cfg = _config(tmp_path, use_ema=True)
    cfg.data.mixup_alpha = 0.2
    cfg.data.cutmix_alpha = 0.2
    trainer = Trainer(cfg, DataModule(cfg.data), progress=False)
    assert trainer.ema is not None
    assert trainer.mixup is not None and trainer.mixup.enabled
    state = trainer.fit(max_steps=3)
    assert state.history


def test_gradient_accumulation(tmp_path: Path) -> None:
    cfg = _config(tmp_path, accumulate_steps=2, grad_clip_norm=1.0)
    trainer = Trainer(cfg, DataModule(cfg.data), progress=False)
    state = trainer.fit(max_steps=2)
    assert state.global_step >= 1


def test_resume_restores_state(tmp_path: Path) -> None:
    cfg = _config(tmp_path, use_ema=True)
    trainer = Trainer(cfg, DataModule(cfg.data), progress=False)
    trainer.fit(max_steps=3)
    ckpt = trainer.output_dir / "last.pt"

    cfg2 = _config(tmp_path, use_ema=True)
    trainer2 = Trainer(cfg2, DataModule(cfg2.data), progress=False)
    trainer2.resume(ckpt)
    assert trainer2.state.global_step == trainer.state.global_step


def test_early_stopping(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path,
        epochs=5,
        early_stopping_patience=1,
        checkpoint_metric="val_loss",
        checkpoint_mode="min",
    )
    trainer = Trainer(cfg, DataModule(cfg.data), progress=False)
    state = trainer.fit()
    # With patience=1 on a tiny run it must stop well before all 5 epochs.
    assert state.epoch < 4


def test_num_classes_reconciled_to_dataset(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.model.num_classes = 3  # deliberately wrong vs. the fake 10-class dataset
    trainer = Trainer(cfg, DataModule(cfg.data), progress=False)
    assert trainer.model.num_classes == 10


@pytest.mark.parametrize("mode,metric", [("max", "val_acc1"), ("min", "val_loss")])
def test_checkpoint_mode(tmp_path: Path, mode: str, metric: str) -> None:
    cfg = _config(tmp_path, checkpoint_metric=metric, checkpoint_mode=mode)
    trainer = Trainer(cfg, DataModule(cfg.data), progress=False)
    trainer.fit(max_steps=2)
    assert (trainer.output_dir / "best.pt").is_file()
