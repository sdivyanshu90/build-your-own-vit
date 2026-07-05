"""HTTP middleware: request IDs, security headers, and rate limiting.

These implement defensive baselines from the OWASP Secure Headers and API
Security guidance without pulling in heavy third-party dependencies.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Security headers applied to every response (OWASP Secure Headers Project).
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}

RequestHandler = Callable[[Request], Awaitable[Response]]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request ID, add security headers, and record a start time.

    The request ID is available as ``request.state.request_id`` and echoed in
    the ``X-Request-ID`` response header for end-to-end tracing.
    """

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.start_time = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        for key, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-client rate limiting for a set of protected paths.

    Uses an in-process sliding window keyed by client IP. This is adequate for a
    single replica; for horizontal scaling place a shared limiter (e.g. an API
    gateway or Redis token bucket) in front. A limit of ``0`` disables it.

    Args:
        app: The ASGI app.
        limit_per_minute: Allowed requests per 60-second window (0 disables).
        protected_prefixes: Only paths starting with one of these are limited.
    """

    def __init__(
        self,
        app: object,
        *,
        limit_per_minute: int,
        protected_prefixes: tuple[str, ...] = ("/v1/predict",),
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.limit = limit_per_minute
        self.window = 60.0
        self.protected_prefixes = protected_prefixes
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _is_protected(self, path: str) -> bool:
        return any(path.startswith(p) for p in self.protected_prefixes)

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        if self.limit <= 0 or not self._is_protected(request.url.path):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        bucket = self._hits[client_ip]
        # Evict timestamps older than the window.
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()
        if len(bucket) >= self.limit:
            retry_after = int(self.window - (now - bucket[0])) + 1
            request_id = getattr(request.state, "request_id", None)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "detail": "Rate limit exceeded. Slow down.",
                    "request_id": request_id,
                },
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)
        return await call_next(request)
