"""The training engine.

:class:`Trainer` wires together the model, data, optimizer, schedule, mixed
precision, regularization (Mixup/CutMix, EMA), checkpointing, and early
stopping into a single ``fit()`` call. It is intentionally framework-light (no
Lightning dependency) so the control flow is fully visible and auditable.

Design highlights:

* **Mixed precision** via :func:`torch.autocast` with a gradient scaler for fp16.
* **Gradient accumulation** to simulate large batches on small GPUs.
* **No-decay parameter grouping** (see :mod:`vit.training.optim`).
* **Atomic, self-describing checkpoints** (best + last) that carry the config.
* **EMA weights** used for validation/serving when enabled.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vit.config import Config
from vit.data.datamodule import DataModule
from vit.models.factory import build_model
from vit.training.ema import ModelEma
from vit.training.metrics import AverageMeter, accuracy
from vit.training.mixup import MixupCutmix, SoftTargetCrossEntropy
from vit.training.optim import build_optimizer
from vit.training.scheduler import build_scheduler
from vit.utils.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from vit.utils.device import autocast_dtype, configure_threads, resolve_device
from vit.utils.logging import get_logger
from vit.utils.seed import seed_everything

logger = get_logger(__name__)


@dataclass
class TrainState:
    """Mutable training bookkeeping."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("-inf")
    epochs_without_improvement: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


class Trainer:
    """Coordinates the end-to-end training lifecycle.

    Args:
        config: The full experiment configuration.
        datamodule: A (not-yet-setup) data module; ``setup()`` is called here.
        model: Optional pre-built model. If ``None``, one is built from the
            config with ``num_classes`` reconciled to the dataset.
        progress: Show tqdm progress bars (disable for tests/CI logs).
    """

    def __init__(
        self,
        config: Config,
        datamodule: DataModule,
        *,
        model: nn.Module | None = None,
        progress: bool = True,
    ) -> None:
        self.config = config
        self.progress = progress
        seed_everything(config.train.seed, deterministic=config.train.deterministic)
        configure_threads(config.train.num_threads)

        self.device = resolve_device(config.train.device)
        self.datamodule = datamodule.setup()
        # Reconcile the head width with the actual dataset for dynamic datasets.
        if config.model.num_classes != self.datamodule.num_classes:
            config.model.num_classes = self.datamodule.num_classes

        self.model = (model or build_model(config.model)).to(self.device)
        if config.train.compile_model:
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        self.optimizer = build_optimizer(self.model, config.optim)
        train_loader = self.datamodule.train_dataloader()
        steps_per_epoch = max(len(train_loader) // config.train.accumulate_steps, 1)
        self.scheduler = build_scheduler(
            self.optimizer,
            config.scheduler,
            steps_per_epoch=steps_per_epoch,
            epochs=config.train.epochs,
            base_lr=config.optim.lr,
        )

        self.amp_dtype = autocast_dtype(self.device, config.train.precision)
        use_scaler = self.amp_dtype == torch.float16 and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

        self.mixup = self._build_mixup()
        self.train_criterion: nn.Module = (
            SoftTargetCrossEntropy()
            if self.mixup and self.mixup.enabled
            else nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing)
        )
        self.eval_criterion = nn.CrossEntropyLoss()

        self.ema = (
            ModelEma(self.model, decay=config.train.ema_decay) if config.train.use_ema else None
        )
        self.state = TrainState()
        self.output_dir = Path(config.train.output_dir) / config.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- setup helpers -----------------------------------------------------
    def _build_mixup(self) -> MixupCutmix | None:
        d = self.config.data
        if d.mixup_alpha <= 0 and d.cutmix_alpha <= 0:
            return None
        return MixupCutmix(
            num_classes=self.datamodule.num_classes,
            mixup_alpha=d.mixup_alpha,
            cutmix_alpha=d.cutmix_alpha,
            label_smoothing=self.config.train.label_smoothing,
            seed=self.config.train.seed,
        )

    # -- public API --------------------------------------------------------
    def fit(self, *, max_steps: int | None = None) -> TrainState:
        """Run the full training loop.

        Args:
            max_steps: Optional hard cap on total optimizer steps (used by smoke
                tests). ``None`` runs the configured number of epochs.

        Returns:
            The final :class:`TrainState`, including per-epoch history and the
            best observed metric.
        """
        val_loader = self.datamodule.val_dataloader()
        logger.info(
            "Starting training",
            extra={
                "run_name": self.config.run_name,
                "device": str(self.device),
                "epochs": self.config.train.epochs,
                "params": sum(p.numel() for p in self.model.parameters()),
            },
        )
        for epoch in range(self.state.epoch, self.config.train.epochs):
            self.state.epoch = epoch
            train_stats = self._train_one_epoch(epoch, max_steps=max_steps)
            metrics = dict(train_stats)
            if val_loader is not None:
                val_stats = self.evaluate(val_loader, use_ema=self.ema is not None)
                metrics.update(val_stats)
            metrics["epoch"] = float(epoch)
            metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
            self.state.history.append(metrics)
            logger.info("Epoch complete", extra=metrics)

            improved = self._update_best(metrics)
            self._save(is_best=improved)
            if self._should_early_stop():
                logger.info(
                    "Early stopping triggered",
                    extra={"epoch": epoch, "patience": self.config.train.early_stopping_patience},
                )
                break
            if max_steps is not None and self.state.global_step >= max_steps:
                break
        return self.state

    def evaluate(
        self, loader: DataLoader, *, use_ema: bool = False, max_batches: int | None = None
    ) -> dict[str, float]:
        """Evaluate on a loader, returning loss / top-1 / top-5 metrics."""
        model = self._eval_model(use_ema)
        model.eval()
        loss_m, acc1_m, acc5_m = AverageMeter(), AverageMeter(), AverageMeter()
        num_classes = self.datamodule.num_classes
        topk = (1, 5) if num_classes >= 5 else (1,)

        with torch.no_grad():
            for i, (images, targets) in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with self._autocast():
                    logits = model(images)
                    loss = self.eval_criterion(logits, targets)
                accs = accuracy(logits, targets, topk=topk)
                bs = images.size(0)
                loss_m.update(loss.item(), bs)
                acc1_m.update(accs[0], bs)
                acc5_m.update(accs[-1] if len(accs) > 1 else accs[0], bs)
        return {
            "val_loss": loss_m.average,
            "val_acc1": acc1_m.average,
            "val_acc5": acc5_m.average,
        }

    # -- training internals ------------------------------------------------
    def _train_one_epoch(self, epoch: int, *, max_steps: int | None) -> dict[str, float]:
        self.model.train()
        loader = self.datamodule.train_dataloader()
        loss_meter = AverageMeter()
        accum = self.config.train.accumulate_steps
        clip = self.config.train.grad_clip_norm
        start = time.perf_counter()

        iterator: Any = loader
        if self.progress:
            iterator = tqdm(loader, desc=f"epoch {epoch}", leave=False)

        self.optimizer.zero_grad(set_to_none=True)
        for step, (images, targets) in enumerate(iterator):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if self.mixup and self.mixup.enabled:
                images, soft_targets = self.mixup(images, targets)
                loss_target: torch.Tensor = soft_targets
            else:
                loss_target = targets

            with self._autocast():
                logits = self.model(images)
                loss = self.train_criterion(logits, loss_target) / accum

            self.scaler.scale(loss).backward()
            loss_meter.update(loss.item() * accum, images.size(0))

            if (step + 1) % accum == 0:
                if clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.state.global_step += 1
                if self.ema is not None:
                    self.ema.update(self.model)
                if self.state.global_step % self.config.train.log_interval == 0:
                    logger.info(
                        "train step",
                        extra={
                            "step": self.state.global_step,
                            "loss": round(loss_meter.average, 4),
                            "lr": self.optimizer.param_groups[0]["lr"],
                        },
                    )
                if max_steps is not None and self.state.global_step >= max_steps:
                    break

        return {
            "train_loss": loss_meter.average,
            "epoch_time_s": time.perf_counter() - start,
        }

    def _eval_model(self, use_ema: bool) -> nn.Module:
        if use_ema and self.ema is not None:
            return self.ema.module
        return self.model

    def _autocast(self) -> Any:
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    # -- checkpointing / stopping -----------------------------------------
    def _update_best(self, metrics: dict[str, float]) -> bool:
        metric_name = self.config.train.checkpoint_metric
        if metric_name not in metrics:
            return False
        value = metrics[metric_name]
        # Normalize to "higher is better" using the configured mode.
        signed = value if self.config.train.checkpoint_mode == "max" else -value
        if signed > self.state.best_metric:
            self.state.best_metric = signed
            self.state.epochs_without_improvement = 0
            return True
        self.state.epochs_without_improvement += 1
        return False

    def _should_early_stop(self) -> bool:
        patience = self.config.train.early_stopping_patience
        return patience > 0 and self.state.epochs_without_improvement >= patience

    def _build_checkpoint(self) -> Checkpoint:
        raw_model = getattr(self.model, "_orig_mod", self.model)  # unwrap compile
        return Checkpoint(
            model_state=raw_model.state_dict(),
            model_config=self.config.model.model_dump(mode="json"),
            class_names=self.datamodule.class_names,
            preprocess={
                "image_size": self.config.data.image_size,
                "mean": list(self.config.data.mean),
                "std": list(self.config.data.std),
            },
            epoch=self.state.epoch,
            global_step=self.state.global_step,
            metrics=self.state.history[-1] if self.state.history else {},
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            scaler_state=self.scaler.state_dict(),
            ema_state=self.ema.state_dict() if self.ema else None,
        )

    def _save(self, *, is_best: bool) -> None:
        ckpt = self._build_checkpoint()
        save_checkpoint(ckpt, self.output_dir / "last.pt")
        if is_best:
            save_checkpoint(ckpt, self.output_dir / "best.pt")
            logger.info(
                "Saved new best checkpoint",
                extra={
                    "metric": self.config.train.checkpoint_metric,
                    "value": self.state.best_metric,
                },
            )

    def resume(self, path: str | Path) -> None:
        """Restore model/optimizer/scheduler/EMA state from a checkpoint."""
        ckpt = load_checkpoint(path, map_location=self.device)
        raw_model = getattr(self.model, "_orig_mod", self.model)
        raw_model.load_state_dict(ckpt.model_state)
        if ckpt.optimizer_state:
            self.optimizer.load_state_dict(ckpt.optimizer_state)
        if ckpt.scheduler_state:
            self.scheduler.load_state_dict(ckpt.scheduler_state)
        if ckpt.scaler_state:
            self.scaler.load_state_dict(ckpt.scaler_state)
        if ckpt.ema_state and self.ema is not None:
            self.ema.load_state_dict(ckpt.ema_state)
        self.state.epoch = ckpt.epoch + 1
        self.state.global_step = ckpt.global_step
        logger.info("Resumed from checkpoint", extra={"path": str(path), "epoch": ckpt.epoch})
