"""Tests for logging, seeding, device, and checkpoint utilities."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest
import torch

from vit.utils.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    Checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from vit.utils.device import autocast_dtype, configure_threads, resolve_device
from vit.utils.logging import JsonFormatter, configure_logging, get_logger
from vit.utils.seed import seed_everything, worker_init_fn


# -- logging ----------------------------------------------------------------
def test_json_formatter_emits_valid_json() -> None:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hi", (), None)
    record.custom_field = "value"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hi"
    assert payload["level"] == "INFO"
    assert payload["custom_field"] == "value"


def test_json_formatter_includes_exception() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "err", (), sys.exc_info())
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError" in payload["exception"]


def test_configure_logging_is_idempotent() -> None:
    stream = io.StringIO()
    configure_logging("DEBUG", json_logs=True, stream=stream)
    configure_logging("INFO", json_logs=False, stream=stream)
    root = logging.getLogger()
    assert len(root.handlers) == 1  # handlers replaced, not stacked


def test_configure_logging_unknown_level_defaults_info() -> None:
    configure_logging("NOTALEVEL")
    assert logging.getLogger().level == logging.INFO


def test_get_logger_returns_named() -> None:
    assert get_logger("vit.test").name == "vit.test"


# -- seed -------------------------------------------------------------------
def test_seed_makes_rng_reproducible() -> None:
    seed_everything(123)
    a = torch.rand(5)
    seed_everything(123)
    b = torch.rand(5)
    torch.testing.assert_close(a, b)


def test_seed_deterministic_flag() -> None:
    assert seed_everything(7, deterministic=True) == 7


def test_seed_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        seed_everything(-1)
    with pytest.raises(ValueError):
        seed_everything(2**32)


def test_worker_init_fn_runs() -> None:
    worker_init_fn(3)  # should not raise


# -- device -----------------------------------------------------------------
def test_resolve_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_resolve_auto_returns_valid_device() -> None:
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_resolve_cuda_falls_back_to_cpu_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda").type == "cpu"


def test_resolve_cuda_specific_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda:0").type == "cuda"


def test_resolve_auto_prefers_mps_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("vit.utils.device._mps_available", lambda: True)
    assert resolve_device("auto").type == "mps"
    assert resolve_device("mps").type == "mps"


def test_resolve_auto_cpu_when_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("vit.utils.device._mps_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_resolve_mps_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vit.utils.device._mps_available", lambda: False)
    assert resolve_device("mps").type == "cpu"


def test_mps_available_returns_bool() -> None:
    from vit.utils.device import _mps_available

    assert isinstance(_mps_available(), bool)


def test_configure_threads_ignores_zero() -> None:
    configure_threads(0)  # no-op, must not raise


def test_configure_threads_sets_positive() -> None:
    original = torch.get_num_threads()
    try:
        configure_threads(2)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(original)


def test_autocast_dtype_matrix() -> None:
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")
    assert autocast_dtype(cpu, "fp32") is None
    assert autocast_dtype(cpu, "bf16") == torch.bfloat16
    assert autocast_dtype(cpu, "fp16") is None  # unsafe on CPU → disabled
    assert autocast_dtype(cuda, "fp16") == torch.float16
    assert autocast_dtype(cuda, "bf16") == torch.bfloat16


# -- checkpoint -------------------------------------------------------------
def _sample_checkpoint() -> Checkpoint:
    return Checkpoint(
        model_state={"w": torch.ones(2, 2)},
        model_config={"embed_dim": 16},
        class_names=["a", "b"],
        preprocess={"image_size": 16, "mean": [0.5], "std": [0.5]},
        epoch=2,
        global_step=10,
        metrics={"val_acc1": 42.0},
    )


def test_checkpoint_save_load_roundtrip(tmp_path: Path) -> None:
    ckpt = _sample_checkpoint()
    path = save_checkpoint(ckpt, tmp_path / "nested" / "ckpt.pt")
    assert path.is_file()
    loaded = load_checkpoint(path)
    assert loaded.epoch == 2
    assert loaded.class_names == ["a", "b"]
    assert loaded.preprocess["image_size"] == 16
    torch.testing.assert_close(loaded.model_state["w"], torch.ones(2, 2))
    assert loaded.format_version == CHECKPOINT_FORMAT_VERSION


def test_load_missing_checkpoint() -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint("/no/such/file.pt")


def test_from_dict_rejects_future_version() -> None:
    with pytest.raises(ValueError, match="newer"):
        Checkpoint.from_dict({"format_version": 999, "model_state": {}, "model_config": {}})


def test_from_dict_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        Checkpoint.from_dict({"format_version": 1, "model_state": {}})


def test_load_rejects_non_dict_payload(tmp_path: Path) -> None:
    path = tmp_path / "raw.pt"
    torch.save([1, 2, 3], path)
    with pytest.raises(ValueError, match="not a dictionary"):
        load_checkpoint(path)


def test_save_is_atomic_no_tmp_left(tmp_path: Path) -> None:
    path = save_checkpoint(_sample_checkpoint(), tmp_path / "c.pt")
    assert not path.with_suffix(".pt.tmp").exists()
