"""Optimizer construction with correct weight-decay grouping.

Weight decay should **not** be applied to biases, normalization parameters, or
the positional embedding / class token — decaying them hurts ViT accuracy. This
module builds two parameter groups accordingly, a detail that is easy to get
wrong and materially affects results.
"""

from __future__ import annotations

from torch import nn, optim

from vit.config import OptimConfig


def _split_params(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """Split parameters into decay / no-decay groups.

    No-decay: any parameter with ``ndim <= 1`` (biases, LayerNorm weights) plus
    the explicitly-named ``pos_embed`` and ``cls_token``.
    """
    decay, no_decay = [], []
    no_decay_names = {"pos_embed", "cls_token"}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        leaf = name.split(".")[-1]
        if param.ndim <= 1 or leaf in no_decay_names:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> optim.Optimizer:
    """Construct an optimizer with proper no-decay grouping.

    Raises:
        ValueError: If ``cfg.name`` is not a supported optimizer.
    """
    groups = _split_params(model, cfg.weight_decay)
    if cfg.name == "adamw":
        return optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
    if cfg.name == "sgd":
        return optim.SGD(
            groups,
            lr=cfg.lr,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov and cfg.momentum > 0,
        )
    raise ValueError(f"Unsupported optimizer {cfg.name!r}.")  # pragma: no cover
