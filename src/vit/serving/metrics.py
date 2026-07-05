"""Prometheus metrics for the inference API.

Metrics are registered on a dedicated :class:`~prometheus_client.CollectorRegistry`
(not the global default) so multiple app instances — e.g. in the test suite —
never collide on duplicate time-series registration.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    """Bundle of application metrics bound to a private registry."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests_total = Counter(
            "vit_requests_total",
            "Total HTTP requests handled.",
            labelnames=("method", "path", "status"),
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "vit_request_latency_seconds",
            "Request latency in seconds.",
            labelnames=("path",),
            registry=self.registry,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )
        self.inference_latency = Histogram(
            "vit_inference_latency_seconds",
            "Model inference latency in seconds.",
            registry=self.registry,
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )
        self.model_loaded = Gauge(
            "vit_model_loaded",
            "1 if a model is loaded and serveable, else 0.",
            registry=self.registry,
        )
        self.predictions_total = Counter(
            "vit_predictions_total",
            "Total successful predictions served.",
            registry=self.registry,
        )
