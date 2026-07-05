"""Structured logging configuration.

The library emits logs through the standard :mod:`logging` module so that host
applications keep full control over handlers and propagation. Two formatters are
provided:

* :class:`JsonFormatter` — one JSON object per line, suitable for log shippers
  (Loki, CloudWatch, Stackdriver) and the Twelve-Factor "logs as event streams"
  principle.
* A human-readable console formatter for local development.

Call :func:`configure_logging` once at process start (the CLI and the serving
app both do this).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects.

    Any non-reserved attribute attached to the record (via ``logger.info(...,
    extra={...})``) is merged into the output, enabling structured context such
    as ``request_id`` or ``epoch`` without string interpolation.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Compact, colorless, human-readable formatter for interactive use."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )


def configure_logging(
    level: str | int = "INFO",
    *,
    json_logs: bool = False,
    stream: Any = None,
) -> None:
    """Configure the root logger idempotently.

    Args:
        level: Logging level name or numeric value.
        json_logs: When ``True`` emit JSON lines, otherwise a console format.
        stream: Output stream (defaults to ``sys.stdout``). Injectable for tests.
    """
    root = logging.getLogger()
    numeric = logging.getLevelName(level) if isinstance(level, str) else level
    if not isinstance(numeric, int):  # unknown level name → sensible default
        numeric = logging.INFO
    root.setLevel(numeric)

    # Replace existing handlers so repeated calls (e.g. in tests) do not stack.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else ConsoleFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Thin wrapper for a consistent import site."""
    return logging.getLogger(name)
