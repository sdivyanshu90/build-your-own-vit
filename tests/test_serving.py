"""Tests for the FastAPI serving layer."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vit.serving import ModelService, ServingSettings, create_app
from vit.serving.settings import ServingSettings as Settings


def _client(settings: ServingSettings) -> TestClient:
    svc = ModelService(settings)
    svc.load()
    app = create_app(settings, service=svc)
    return TestClient(app)


@pytest.fixture
def ready_client(trained_checkpoint: Path) -> TestClient:
    settings = Settings(
        model_checkpoint=str(trained_checkpoint),
        device="cpu",
        rate_limit_rpm=0,
        enable_metrics=True,
    )
    return _client(settings)


@pytest.fixture
def png_upload(png_bytes: bytes) -> dict[str, tuple[str, io.BytesIO, str]]:
    return {"file": ("img.png", io.BytesIO(png_bytes), "image/png")}


# -- health / metadata ------------------------------------------------------
def test_root(ready_client: TestClient) -> None:
    body = ready_client.get("/").json()
    assert body["name"] == "build-your-own-vit"


def test_healthz(ready_client: TestClient) -> None:
    body = ready_client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_readyz_ready(ready_client: TestClient) -> None:
    assert ready_client.get("/readyz").status_code == 200


def test_metadata(ready_client: TestClient) -> None:
    body = ready_client.get("/v1/metadata").json()
    assert body["num_classes"] == 10
    assert body["image_size"] == 16
    assert len(body["class_names"]) == 10


def test_security_headers_present(ready_client: TestClient) -> None:
    resp = ready_client.get("/healthz")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert "X-Request-ID" in resp.headers


# -- predict ----------------------------------------------------------------
def test_predict_success(ready_client: TestClient, png_upload: dict) -> None:
    resp = ready_client.post("/v1/predict?top_k=3", files=png_upload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["top_k"] == 3
    assert len(body["predictions"]) == 3
    assert body["latency_ms"] >= 0
    assert body["request_id"]


def test_predict_unsupported_media_type(ready_client: TestClient) -> None:
    files = {"file": ("x.txt", io.BytesIO(b"hello"), "text/plain")}
    resp = ready_client.post("/v1/predict", files=files)
    assert resp.status_code == 415
    assert resp.json()["error"] == "unsupported_media_type"


def test_predict_corrupt_image(ready_client: TestClient) -> None:
    files = {"file": ("x.png", io.BytesIO(b"not-a-real-png"), "image/png")}
    resp = ready_client.post("/v1/predict", files=files)
    assert resp.status_code == 400


def test_predict_empty_upload(ready_client: TestClient) -> None:
    files = {"file": ("x.png", io.BytesIO(b""), "image/png")}
    resp = ready_client.post("/v1/predict", files=files)
    assert resp.status_code == 400


def test_predict_payload_too_large(trained_checkpoint: Path, png_bytes: bytes) -> None:
    settings = Settings(
        model_checkpoint=str(trained_checkpoint),
        device="cpu",
        rate_limit_rpm=0,
        max_upload_bytes=10,
    )
    client = _client(settings)
    files = {"file": ("img.png", io.BytesIO(png_bytes), "image/png")}
    resp = client.post("/v1/predict", files=files)
    assert resp.status_code == 413
    assert resp.json()["error"] == "payload_too_large"


# -- readiness / errors -----------------------------------------------------
def test_not_ready_returns_503() -> None:
    settings = Settings(model_checkpoint=None, device="cpu", rate_limit_rpm=0)
    client = _client(settings)
    assert client.get("/readyz").status_code == 503
    img = io.BytesIO()
    Image.new("RGB", (16, 16)).save(img, format="PNG")
    img.seek(0)
    resp = client.post("/v1/predict", files={"file": ("i.png", img, "image/png")})
    assert resp.status_code == 503


# -- auth -------------------------------------------------------------------
def test_auth_required_when_token_set(trained_checkpoint: Path, png_bytes: bytes) -> None:
    settings = Settings(
        model_checkpoint=str(trained_checkpoint),
        device="cpu",
        rate_limit_rpm=0,
        api_token="s3cret",
    )
    client = _client(settings)
    files = {"file": ("img.png", io.BytesIO(png_bytes), "image/png")}
    # No token → 401
    assert client.post("/v1/predict", files=files).status_code == 401
    # Wrong token → 401
    files = {"file": ("img.png", io.BytesIO(png_bytes), "image/png")}
    bad = client.post("/v1/predict", files=files, headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401
    # Correct token → 200
    files = {"file": ("img.png", io.BytesIO(png_bytes), "image/png")}
    ok = client.post("/v1/predict", files=files, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


# -- rate limiting ----------------------------------------------------------
def test_rate_limiting(trained_checkpoint: Path, png_bytes: bytes) -> None:
    settings = Settings(
        model_checkpoint=str(trained_checkpoint),
        device="cpu",
        rate_limit_rpm=2,
    )
    client = _client(settings)

    def _post() -> int:
        files = {"file": ("img.png", io.BytesIO(png_bytes), "image/png")}
        return client.post("/v1/predict", files=files).status_code

    assert _post() == 200
    assert _post() == 200
    resp_code = _post()
    assert resp_code == 429


# -- metrics ----------------------------------------------------------------
def test_metrics_endpoint(ready_client: TestClient, png_upload: dict) -> None:
    ready_client.post("/v1/predict", files=png_upload)
    resp = ready_client.get("/metrics")
    assert resp.status_code == 200
    assert "vit_predictions_total" in resp.text
    assert "vit_model_loaded" in resp.text


def test_metrics_disabled(trained_checkpoint: Path) -> None:
    settings = Settings(
        model_checkpoint=str(trained_checkpoint),
        device="cpu",
        rate_limit_rpm=0,
        enable_metrics=False,
    )
    client = _client(settings)
    assert client.get("/metrics").status_code == 404


# -- service ----------------------------------------------------------------
def test_service_load_error_no_checkpoint() -> None:
    svc = ModelService(Settings(model_checkpoint=None))
    svc.load()
    assert not svc.is_ready
    assert svc.load_error is not None


def test_service_load_error_bad_path(tmp_path: Path) -> None:
    svc = ModelService(Settings(model_checkpoint=str(tmp_path / "missing.pt")))
    svc.load()
    assert not svc.is_ready
    assert "FileNotFoundError" in (svc.load_error or "")


def test_service_predictor_raises_when_unloaded() -> None:
    svc = ModelService(Settings(model_checkpoint=None))
    with pytest.raises(RuntimeError):
        _ = svc.predictor


def test_app_autoloads_model_in_lifespan(trained_checkpoint: Path, png_bytes: bytes) -> None:
    # No injected service: the lifespan startup must load the model itself
    # (the real production path).
    settings = Settings(
        model_checkpoint=str(trained_checkpoint),
        device="cpu",
        rate_limit_rpm=0,
    )
    app = create_app(settings)
    with TestClient(app) as client:  # context manager triggers lifespan
        assert client.get("/readyz").status_code == 200
        files = {"file": ("img.png", io.BytesIO(png_bytes), "image/png")}
        assert client.post("/v1/predict", files=files).status_code == 200
