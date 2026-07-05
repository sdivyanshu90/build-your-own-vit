"""Evaluation metrics and running aggregators."""

from __future__ import annotations

import torch


class AverageMeter:
    """Tracks a running average, e.g. loss or accuracy over a validation pass.

    Example:
        >>> m = AverageMeter()
        >>> m.update(1.0, n=2); m.update(4.0, n=1)
        >>> round(m.average, 4)
        2.0
    """

    __slots__ = ("count", "sum")

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        """Add ``value`` observed ``n`` times."""
        if n < 0:
            raise ValueError("n must be non-negative.")
        self.sum += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        return self.sum / self.count if self.count else 0.0


@torch.no_grad()
def accuracy(
    logits: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...] = (1,)
) -> list[float]:
    """Compute top-k accuracy (as percentages) for the given ``k`` values.

    Args:
        logits: Model outputs of shape ``(B, num_classes)``.
        targets: Ground-truth class indices of shape ``(B,)``.
        topk: The ``k`` values to compute accuracy for.

    Returns:
        One accuracy percentage per entry in ``topk``.

    Raises:
        ValueError: If shapes are incompatible or any ``k`` exceeds the number
            of classes.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (B, C), got {tuple(logits.shape)}.")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must be 1-D with the same batch size as logits.")
    num_classes = logits.shape[1]
    maxk = max(topk)
    if maxk > num_classes:
        raise ValueError(f"topk={topk} exceeds number of classes ({num_classes}).")

    batch_size = targets.shape[0]
    if batch_size == 0:
        return [0.0 for _ in topk]

    # (B, maxk) indices of the highest-scoring classes, then compare to target.
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # (maxk, B)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results: list[float] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum().item()
        results.append(correct_k * 100.0 / batch_size)
    return results
