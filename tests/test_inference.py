"""Tests for the predictor and ONNX export."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from vit.inference import Predictor, export_checkpoint_to_onnx, export_onnx
from vit.inference.predictor import Prediction


def test_predictor_from_checkpoint(trained_checkpoint: Path, rgb_image: Image.Image) -> None:
    predictor = Predictor.from_checkpoint(trained_checkpoint, device="cpu")
    results = predictor.predict(rgb_image, top_k=3)
    assert len(results) == 1
    assert len(results[0]) == 3
    assert all(isinstance(p, Prediction) for p in results[0])
    # Probabilities are sorted descending and in [0, 1].
    probs = [p.probability for p in results[0]]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_predictor_batch(trained_checkpoint: Path, rgb_image: Image.Image) -> None:
    predictor = Predictor.from_checkpoint(trained_checkpoint, device="cpu")
    results = predictor.predict([rgb_image, rgb_image], top_k=2)
    assert len(results) == 2


def test_predict_proba_sums_to_one(trained_checkpoint: Path, rgb_image: Image.Image) -> None:
    predictor = Predictor.from_checkpoint(trained_checkpoint, device="cpu")
    probs = predictor.predict_proba(rgb_image)
    assert probs.shape == (1, 10)
    torch.testing.assert_close(probs.sum(-1), torch.ones(1), rtol=1e-4, atol=1e-4)


def test_predict_empty_list_raises(trained_checkpoint: Path) -> None:
    predictor = Predictor.from_checkpoint(trained_checkpoint, device="cpu")
    with pytest.raises(ValueError, match="No images"):
        predictor.predict([])


def test_top_k_capped_to_num_classes(trained_checkpoint: Path, rgb_image: Image.Image) -> None:
    predictor = Predictor.from_checkpoint(trained_checkpoint, device="cpu")
    results = predictor.predict(rgb_image, top_k=999)
    assert len(results[0]) == 10  # capped at num_classes


def test_predictor_uses_ema_when_available(
    trained_checkpoint: Path, rgb_image: Image.Image
) -> None:
    # Both should load and predict; use_ema just selects the weight set.
    p_ema = Predictor.from_checkpoint(trained_checkpoint, device="cpu", use_ema=True)
    p_raw = Predictor.from_checkpoint(trained_checkpoint, device="cpu", use_ema=False)
    assert p_ema.predict(rgb_image)[0][0].index >= 0
    assert p_raw.predict(rgb_image)[0][0].index >= 0


def test_export_onnx_from_model(tmp_path: Path, tiny_model_config) -> None:  # type: ignore[no-untyped-def]
    from vit.models import build_model

    onnx = pytest.importorskip("onnx")
    model = build_model(tiny_model_config)
    out = export_onnx(model, tmp_path / "m.onnx", image_size=16)
    assert out.is_file()
    onnx.checker.check_model(onnx.load(str(out)))


def test_export_checkpoint_to_onnx(tmp_path: Path, trained_checkpoint: Path) -> None:
    onnx = pytest.importorskip("onnx")
    out = export_checkpoint_to_onnx(trained_checkpoint, tmp_path / "c.onnx")
    assert out.is_file()
    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    input_names = [i.name for i in model.graph.input]
    assert "input" in input_names
