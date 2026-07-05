"""Pydantic request/response schemas for the inference API.

These models double as validation and as the source of the OpenAPI schema that
FastAPI serves at ``/docs`` and ``/openapi.json``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """A single ranked class prediction."""

    label: str = Field(description="Human-readable class label.")
    index: int = Field(description="Class index in the model's output space.")
    probability: float = Field(ge=0.0, le=1.0, description="Softmax probability for this class.")


class PredictResponse(BaseModel):
    """Response body for a successful prediction."""

    request_id: str = Field(description="Correlates the response with server logs.")
    model_version: str = Field(description="Library version that served the request.")
    top_k: int = Field(description="Number of predictions returned.")
    predictions: list[PredictionItem] = Field(
        description="Predictions sorted by descending probability."
    )
    latency_ms: float = Field(description="Server-side inference latency.")


class HealthResponse(BaseModel):
    """Liveness/readiness probe payload."""

    status: str = Field(description="'ok' or 'degraded'.")
    model_loaded: bool = Field(description="Whether a model is loaded and serveable.")
    version: str = Field(description="Library version.")


class MetadataResponse(BaseModel):
    """Static model metadata for clients."""

    version: str
    model_loaded: bool
    num_classes: int
    class_names: list[str]
    image_size: int
    device: str


class ErrorResponse(BaseModel):
    """Uniform error envelope returned for all handled failures."""

    error: str = Field(description="Stable, machine-readable error code.")
    detail: str = Field(description="Human-readable explanation.")
    request_id: str | None = Field(default=None)
