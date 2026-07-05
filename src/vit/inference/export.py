"""Model export to ONNX for portable, framework-agnostic serving.

Exporting to ONNX lets the model run under ONNX Runtime, TensorRT, or other
accelerators without a PyTorch dependency. The exported graph uses a dynamic
batch axis so a single artifact serves any batch size.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from vit.config import ModelConfig
from vit.models.factory import build_model
from vit.utils.checkpoint import load_checkpoint
from vit.utils.logging import get_logger

logger = get_logger(__name__)


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    *,
    image_size: int,
    in_channels: int = 3,
    opset: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export a model to ONNX with a dynamic batch dimension.

    Args:
        model: The (eval-mode) model to export.
        output_path: Destination ``.onnx`` path.
        image_size: Spatial size of the dummy input.
        in_channels: Input channels for the dummy input.
        opset: ONNX opset version.
        dynamic_batch: Mark the batch axis as dynamic.

    Returns:
        The path the model was written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    dummy = torch.randn(1, in_channels, image_size, image_size)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    logger.info("Exported ONNX model", extra={"path": str(output_path), "opset": opset})
    return output_path


def export_checkpoint_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    use_ema: bool = True,
    opset: int = 17,
) -> Path:
    """Load a checkpoint and export its model to ONNX in one call."""
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    model_cfg = ModelConfig.model_validate(ckpt.model_config)
    model = build_model(model_cfg)
    state = ckpt.ema_state if (use_ema and ckpt.ema_state) else ckpt.model_state
    model.load_state_dict(state)
    return export_onnx(
        model,
        output_path,
        image_size=model_cfg.image_size,
        in_channels=model_cfg.in_channels,
        opset=opset,
    )
