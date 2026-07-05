"""Model lifecycle management for the serving layer.

:class:`ModelService` owns the :class:`~vit.inference.predictor.Predictor` and
tracks readiness. Loading is decoupled from process start so the container can
report *liveness* immediately (the process is up) while *readiness* flips to
true only once the model is actually serveable — the standard Kubernetes
liveness/readiness split.
"""

from __future__ import annotations

import threading

from PIL import Image

from vit.inference.predictor import Prediction, Predictor
from vit.serving.settings import ServingSettings
from vit.utils.device import configure_threads
from vit.utils.logging import get_logger

logger = get_logger(__name__)


class ModelService:
    """Thread-safe holder for the inference model and its readiness state."""

    def __init__(self, settings: ServingSettings) -> None:
        self.settings = settings
        self._predictor: Predictor | None = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        """Attempt to load the model. Never raises — records errors for /readyz."""
        configure_threads(self.settings.num_threads)
        if not self.settings.model_checkpoint:
            self._load_error = "No VIT_MODEL_CHECKPOINT configured."
            logger.warning(self._load_error)
            return
        try:
            with self._lock:
                self._predictor = Predictor.from_checkpoint(
                    self.settings.model_checkpoint,
                    device=self.settings.device,
                    use_ema=self.settings.use_ema,
                )
            self._load_error = None
            logger.info(
                "Model loaded",
                extra={
                    "checkpoint": self.settings.model_checkpoint,
                    "num_classes": len(self._predictor.class_names),
                    "device": str(self._predictor.device),
                },
            )
        except Exception as exc:
            self._predictor = None
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.error("Model load failed", extra={"error": self._load_error})

    @property
    def is_ready(self) -> bool:
        return self._predictor is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def predictor(self) -> Predictor:
        if self._predictor is None:
            raise RuntimeError("Model is not loaded.")
        return self._predictor

    # -- inference ---------------------------------------------------------
    def predict(self, image: Image.Image, *, top_k: int) -> list[Prediction]:
        """Run a single-image prediction. Returns the ranked predictions."""
        return self.predictor.predict(image, top_k=top_k)[0]

    # -- metadata ----------------------------------------------------------
    def metadata(self) -> dict[str, object]:
        if self._predictor is None:
            return {
                "model_loaded": False,
                "num_classes": 0,
                "class_names": [],
                "image_size": 0,
                "device": self.settings.device,
            }
        return {
            "model_loaded": True,
            "num_classes": len(self._predictor.class_names),
            "class_names": self._predictor.class_names,
            "image_size": self._predictor.image_size,
            "device": str(self._predictor.device),
        }
