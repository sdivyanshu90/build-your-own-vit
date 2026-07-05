"""Device selection and runtime tuning helpers."""

from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a device string into a concrete :class:`torch.device`.

    Args:
        preference: One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:N"`` or
            ``"mps"``. ``"auto"`` prefers CUDA, then Apple MPS, then CPU.

    Returns:
        A concrete device. When a specific accelerator is requested but
        unavailable, the function falls back to CPU rather than raising, so that
        the same config runs on any machine.
    """
    pref = preference.strip().lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if pref.startswith("cuda"):
        return torch.device(pref) if torch.cuda.is_available() else torch.device("cpu")
    if pref == "mps":
        return torch.device("mps") if _mps_available() else torch.device("cpu")
    return torch.device("cpu")


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def configure_threads(num_threads: int) -> None:
    """Set the intra-op thread count for CPU inference/training.

    A value of ``0`` (or negative) leaves the PyTorch default in place. Pinning
    threads is important in containers where the host reports many more cores
    than the cgroup actually grants.
    """
    if num_threads and num_threads > 0:
        torch.set_num_threads(num_threads)


def autocast_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    """Map a precision string to an autocast dtype for the given device.

    Returns ``None`` when mixed precision should be disabled (e.g. ``"fp32"`` or
    autocast unsupported on the device).
    """
    precision = precision.lower()
    if precision in {"fp32", "32", "float32"}:
        return None
    if device.type == "cuda":
        if precision in {"bf16", "bfloat16"}:
            return torch.bfloat16
        return torch.float16
    if device.type == "cpu" and precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    # No safe autocast path (e.g. fp16 on CPU) → run in full precision.
    return None
