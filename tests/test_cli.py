"""Tests for the command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from vit.cli import _parse_overrides, build_parser, main


def _write_config(tmp_path: Path) -> Path:
    payload = {
        "run_name": "cli-test",
        "model": {
            "image_size": 16,
            "patch_size": 4,
            "num_classes": 10,
            "embed_dim": 16,
            "depth": 1,
            "num_heads": 2,
            "mlp_ratio": 2.0,
        },
        "data": {"dataset": "fake", "image_size": 16, "batch_size": 8, "num_workers": 0},
        "scheduler": {"warmup_epochs": 0},
        "train": {
            "epochs": 1,
            "precision": "fp32",
            "device": "cpu",
            "output_dir": str(tmp_path / "out"),
            "use_ema": True,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


# -- parsing ----------------------------------------------------------------
def test_parse_overrides_types() -> None:
    ov = _parse_overrides(["train.epochs=5", "optim.lr=1e-3", "train.use_ema=true"])
    assert ov == {"train.epochs": 5, "optim.lr": 1e-3, "train.use_ema": True}


def test_parse_overrides_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid"):
        _parse_overrides(["noequals"])


def test_parser_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


# -- info / init-config -----------------------------------------------------
def test_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "vit version" in out


def test_init_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "gen.yaml"
    code = main(
        [
            "init-config",
            "--preset",
            "vit_tiny",
            "--image-size",
            "16",
            "--patch-size",
            "4",
            "--num-classes",
            "10",
            "-o",
            str(out),
        ]
    )
    assert code == 0
    assert out.is_file()
    cfg = yaml.safe_load(out.read_text())
    assert cfg["model"]["preset"] == "vit_tiny"


# -- full lifecycle ---------------------------------------------------------
def test_train_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _write_config(tmp_path)
    assert main(["train", "--config", str(cfg), "--smoke"]) == 0
    assert (tmp_path / "out" / "cli-test" / "best.pt").is_file()


def test_train_evaluate_predict_export(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _write_config(tmp_path)
    main(["train", "--config", str(cfg), "--smoke"])
    ckpt = tmp_path / "out" / "cli-test" / "best.pt"
    assert ckpt.is_file()

    # evaluate
    assert (
        main(["evaluate", "--config", str(cfg), "--checkpoint", str(ckpt), "--split", "test"]) == 0
    )
    assert "val_acc1" in capsys.readouterr().out

    # predict
    img_path = tmp_path / "img.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(img_path)
    assert (
        main(["predict", "--checkpoint", str(ckpt), "--image", str(img_path), "--top-k", "3"]) == 0
    )
    assert "Predictions" in capsys.readouterr().out

    # export
    onnx_out = tmp_path / "model.onnx"
    assert main(["export", "--checkpoint", str(ckpt), "--output", str(onnx_out)]) == 0
    assert onnx_out.is_file()


def test_train_with_set_overrides(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    code = main(["train", "--config", str(cfg), "--smoke", "--set", "optim.lr=0.0005"])
    assert code == 0


# -- error handling ---------------------------------------------------------
def test_missing_config_returns_1(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["train", "--config", "/nope/missing.yaml"])
    assert code == 1
    assert "Error:" in capsys.readouterr().err


def test_evaluate_split_without_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_config(tmp_path)
    main(["train", "--config", str(cfg), "--smoke"])
    ckpt = tmp_path / "out" / "cli-test" / "best.pt"
    # Force the test loader to be absent to exercise the "no split" branch.
    from vit.data.datamodule import DataModule

    monkeypatch.setattr(DataModule, "test_dataloader", lambda self: None)
    code = main(["evaluate", "--config", str(cfg), "--checkpoint", str(ckpt), "--split", "test"])
    assert code == 2


# -- serve (mocked) ---------------------------------------------------------
def test_serve_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_run(app: str, **kwargs: object) -> None:
        calls["app"] = app
        calls.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    code = main(["serve", "--host", "127.0.0.1", "--port", "9999"])
    assert code == 0
    assert calls["app"] == "vit.serving.app:create_app"
    assert calls["port"] == 9999
