"""FastAPI application for ViT inference.

The app is created by a factory (:func:`create_app`) so tests can inject a
pre-loaded :class:`~vit.serving.service.ModelService` and settings without
touching the environment. Endpoints:

======================  ======  =====================================
Path                    Method  Purpose
======================  ======  =====================================
``/``                   GET     Service banner + links.
``/healthz``            GET     Liveness probe (always 200 if process up).
``/readyz``             GET     Readiness probe (503 until model loaded).
``/v1/metadata``        GET     Model metadata (classes, image size, ...).
``/v1/predict``         POST    Classify an uploaded image.
``/metrics``            GET     Prometheus exposition (if enabled).
======================  ======  =====================================
"""

from __future__ import annotations

import hmac
import io
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from vit import __version__
from vit.serving.metrics import Metrics
from vit.serving.middleware import RateLimitMiddleware, RequestContextMiddleware
from vit.serving.schemas import (
    ErrorResponse,
    HealthResponse,
    MetadataResponse,
    PredictionItem,
    PredictResponse,
)
from vit.serving.service import ModelService
from vit.serving.settings import ServingSettings
from vit.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def create_app(
    settings: ServingSettings | None = None,
    *,
    service: ModelService | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Serving settings. Loaded from the environment if omitted.
        service: Pre-built model service (mainly for tests). If omitted, one is
            created and its ``load()`` runs during the lifespan startup.
    """
    settings = settings or ServingSettings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    metrics = Metrics()
    svc = service or ModelService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:  # only auto-load if the caller didn't inject one
            svc.load()
        metrics.model_loaded.set(1 if svc.is_ready else 0)
        logger.info("Serving app started", extra={"ready": svc.is_ready})
        yield
        logger.info("Serving app shutting down")

    app = FastAPI(
        title="Build Your Own ViT — Inference API",
        version=__version__,
        description="Image classification with a from-scratch Vision Transformer.",
        root_path=settings.root_path,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.service = svc
    app.state.metrics = metrics

    _register_middleware(app, settings, metrics)
    _register_error_handlers(app)
    _register_routes(app, settings, svc, metrics)
    return app


# ---------------------------------------------------------------------------
# Middleware / errors
# ---------------------------------------------------------------------------
def _register_middleware(app: FastAPI, settings: ServingSettings, metrics: Metrics) -> None:
    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    app.add_middleware(RateLimitMiddleware, limit_per_minute=settings.rate_limit_rpm)
    app.add_middleware(RequestContextMiddleware)

    @app.middleware("http")
    async def _record_metrics(request: Request, call_next):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        response = await call_next(request)
        if settings.enable_metrics:
            # Prefer the matched route template (bounded label cardinality) over
            # the raw URL path, which could contain unbounded values.
            route = request.scope.get("route")
            path = route.path if route is not None else request.url.path
            metrics.request_latency.labels(path=path).observe(time.perf_counter() - start)
            metrics.requests_total.labels(
                method=request.method, path=path, status=str(response.status_code)
            ).inc()
        return response


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http_exc(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        payload = ErrorResponse(
            error=_error_code(exc.status_code),
            detail=str(exc.detail),
            request_id=request_id,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled server error", extra={"request_id": request_id})
        payload = ErrorResponse(
            error="internal_error",
            detail="An unexpected error occurred.",
            request_id=request_id,
        )
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, content=payload.model_dump()
        )


def _error_code(status_code: int) -> str:
    return {
        HTTP_400_BAD_REQUEST: "bad_request",
        HTTP_401_UNAUTHORIZED: "unauthorized",
        HTTP_413_REQUEST_ENTITY_TOO_LARGE: "payload_too_large",
        HTTP_415_UNSUPPORTED_MEDIA_TYPE: "unsupported_media_type",
        HTTP_503_SERVICE_UNAVAILABLE: "service_unavailable",
    }.get(status_code, "error")


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------
def _make_auth_dependency(settings: ServingSettings):  # type: ignore[no-untyped-def]
    async def verify_token(request: Request) -> None:
        if not settings.api_token:
            return  # auth disabled
        header = request.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token, settings.api_token):
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid bearer token.",
            )

    return verify_token


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def _register_routes(
    app: FastAPI,
    settings: ServingSettings,
    svc: ModelService,
    metrics: Metrics,
) -> None:
    auth = _make_auth_dependency(settings)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "name": "build-your-own-vit",
            "version": __version__,
            "docs": "/docs",
            "health": "/healthz",
        }

    @app.get("/healthz", response_model=HealthResponse, tags=["health"])
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=svc.is_ready, version=__version__)

    @app.get(
        "/readyz",
        response_model=HealthResponse,
        tags=["health"],
        responses={503: {"model": ErrorResponse}},
    )
    async def readyz() -> Response:
        if svc.is_ready:
            body = HealthResponse(status="ok", model_loaded=True, version=__version__)
            return JSONResponse(content=body.model_dump())
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail=svc.load_error or "Model not loaded.",
        )

    @app.get("/v1/metadata", response_model=MetadataResponse, tags=["model"])
    async def metadata() -> MetadataResponse:
        meta = svc.metadata()
        return MetadataResponse(version=__version__, **meta)

    @app.post(
        "/v1/predict",
        response_model=PredictResponse,
        tags=["inference"],
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def predict(
        request: Request,
        file: UploadFile = File(..., description="Image file to classify."),
        top_k: int = Query(default=settings.default_top_k, ge=1, le=100),
        _: None = Depends(auth),
    ) -> PredictResponse:
        if not svc.is_ready:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail=svc.load_error or "Model not loaded.",
            )
        image = await _read_image(file, settings.max_upload_bytes)

        infer_start = time.perf_counter()
        predictions = svc.predict(image, top_k=top_k)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0
        if settings.enable_metrics:
            metrics.inference_latency.observe(infer_ms / 1000.0)
            metrics.predictions_total.inc()

        return PredictResponse(
            request_id=getattr(request.state, "request_id", ""),
            model_version=__version__,
            top_k=len(predictions),
            predictions=[
                PredictionItem(label=p.label, index=p.index, probability=p.probability)
                for p in predictions
            ],
            latency_ms=round(infer_ms, 3),
        )

    if settings.enable_metrics:

        @app.get("/metrics", include_in_schema=False)
        async def prometheus_metrics() -> Response:
            metrics.model_loaded.set(1 if svc.is_ready else 0)
            return Response(
                content=generate_latest(metrics.registry),
                media_type=CONTENT_TYPE_LATEST,
            )


async def _read_image(file: UploadFile, max_bytes: int) -> Image.Image:
    """Read, size-check, and decode an uploaded image.

    Raises:
        HTTPException: 415 for non-image content types, 413 when the payload
            exceeds ``max_bytes``, 400 when the bytes cannot be decoded.
    """
    if file.content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type {file.content_type!r}. "
            f"Allowed: {sorted(_ALLOWED_IMAGE_TYPES)}.",
        )
    # Read with a hard cap: one extra byte reveals an over-limit payload.
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {max_bytes}-byte limit.",
        )
    if not data:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Empty upload.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Could not decode the uploaded image.",
        ) from exc
    return image.convert("RGB")
