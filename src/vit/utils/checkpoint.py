"""Checkpoint serialization utilities.

A checkpoint is a self-describing bundle that carries everything needed to
resume training or to rebuild a model for inference: the weights, the exact
model configuration, the class-name mapping, and training bookkeeping
(optimizer/scheduler/scaler/EMA state, epoch, step, metrics).

Storing the *config* alongside the *weights* means inference never has to guess
architecture hyper-parameters — the model is reconstructed deterministically
from the file itself.

.. warning::
   ``torch.load`` uses ``pickle`` under the hood. Only load checkpoints from
   sources you trust. This project stores plain Python types plus tensors, but
   the loader still executes the trust boundary of unpickling. See
   ``docs/security.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from vit import __version__

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(slots=True)
class Checkpoint:
    """In-memory representation of a checkpoint bundle."""

    model_state: dict[str, torch.Tensor]
    model_config: dict[str, Any]
    class_names: list[str] | None = None
    preprocess: dict[str, Any] | None = None
    epoch: int = 0
    global_step: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    scaler_state: dict[str, Any] | None = None
    ema_state: dict[str, torch.Tensor] | None = None
    format_version: int = CHECKPOINT_FORMAT_VERSION
    library_version: str = __version__
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "library_version": self.library_version,
            "created_at": self.created_at or datetime.now(tz=timezone.utc).isoformat(),
            "model_state": self.model_state,
            "model_config": self.model_config,
            "class_names": self.class_names,
            "preprocess": self.preprocess,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "metrics": self.metrics,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "scaler_state": self.scaler_state,
            "ema_state": self.ema_state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Checkpoint:
        version = payload.get("format_version", 0)
        if version > CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Checkpoint format version {version} is newer than supported "
                f"version {CHECKPOINT_FORMAT_VERSION}. Upgrade the vit package."
            )
        if "model_state" not in payload or "model_config" not in payload:
            raise ValueError("Malformed checkpoint: missing 'model_state' or 'model_config'.")
        return cls(
            model_state=payload["model_state"],
            model_config=payload["model_config"],
            class_names=payload.get("class_names"),
            preprocess=payload.get("preprocess"),
            epoch=payload.get("epoch", 0),
            global_step=payload.get("global_step", 0),
            metrics=payload.get("metrics", {}) or {},
            optimizer_state=payload.get("optimizer_state"),
            scheduler_state=payload.get("scheduler_state"),
            scaler_state=payload.get("scaler_state"),
            ema_state=payload.get("ema_state"),
            format_version=version or CHECKPOINT_FORMAT_VERSION,
            library_version=payload.get("library_version", "unknown"),
            created_at=payload.get("created_at", ""),
        )


def save_checkpoint(checkpoint: Checkpoint, path: str | Path) -> Path:
    """Atomically persist a checkpoint to ``path``.

    Writes to a temporary sibling file first and then renames it, so a crash
    mid-write can never corrupt an existing good checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint.to_dict(), tmp)
    tmp.replace(path)  # atomic on POSIX within the same filesystem
    return path


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Checkpoint:
    """Load a checkpoint bundle from disk.

    Args:
        path: Path to a ``.pt`` file produced by :func:`save_checkpoint`.
        map_location: Where tensors are materialized (default CPU, which is
            always safe and lets callers move the model afterwards).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not a recognizable checkpoint.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint at {path} is not a dictionary payload.")
    return Checkpoint.from_dict(payload)
