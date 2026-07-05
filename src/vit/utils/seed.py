"""Reproducibility helpers.

Deterministic runs are essential for debugging, regression tests, and
scientific comparison. :func:`seed_everything` seeds every RNG the training and
inference paths touch, and optionally forces cuDNN into deterministic mode.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch RNGs.

    Args:
        seed: The base seed. Must be a non-negative 32-bit integer.
        deterministic: When ``True``, configure cuDNN and PyTorch algorithms for
            deterministic behavior. This can reduce throughput and is intended
            for tests and exact-repro experiments rather than large training
            runs.

    Returns:
        The seed that was applied (echoed for logging convenience).

    Raises:
        ValueError: If ``seed`` is negative or does not fit in 32 bits.
    """
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"seed must be in [0, 2**32-1], got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cudnn.benchmark trades determinism for autotuned kernels; disable it.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # warn_only avoids hard failures on ops without a deterministic kernel.
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return seed


def worker_init_fn(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` giving each worker a distinct, stable seed.

    Derives per-worker seeds from PyTorch's initial seed so that shuffling and
    augmentation remain reproducible across runs while staying decorrelated
    across workers.
    """
    base_seed = torch.initial_seed() % (2**32)
    seed = (base_seed + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
