# syntax=docker/dockerfile:1.7
# =============================================================================
# Multi-stage build for the ViT inference service.
#
# Stage 1 (builder): install dependencies into a virtualenv.
# Stage 2 (runtime): copy the venv into a slim, non-root image.
#
# The result is a small image that runs the FastAPI server as an unprivileged
# user with a healthcheck. CPU wheels are used by default; for GPU serving,
# swap the base image for an nvidia/cuda runtime and install the matching torch.
# =============================================================================

ARG PYTHON_VERSION=3.11

# ----------------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Create an isolated virtualenv we can copy wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU-only torch first (smaller, no CUDA) then the package.
# Copy only metadata first to maximize Docker layer caching.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip setuptools wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.1.0" "torchvision>=0.16.0" \
    && pip install ".[serve,export]"

# ----------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="build-your-own-vit" \
      org.opencontainers.image.description="Vision Transformer inference API" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/your-org/build-your-own-vit"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIT_HOST=0.0.0.0 \
    VIT_PORT=8080

# Minimal OS deps for Pillow (libjpeg/zlib are already present in slim, but
# libgomp is needed by torch). curl is used by the container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user.
RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app configs ./configs

USER app
EXPOSE 8080

# Liveness check hits the process-up probe (does not require a loaded model).
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${VIT_PORT}/healthz" || exit 1

# Run the ASGI app via the CLI's serve command (reads VIT_* env vars).
ENTRYPOINT ["python", "-m", "vit"]
CMD ["serve"]
