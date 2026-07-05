# API Reference

The inference service is a FastAPI application. When running, interactive docs
are available at **`/docs`** (Swagger UI) and **`/redoc`**, and the machine
schema at **`/openapi.json`**.

- Base URL (local): `http://localhost:8080`
- Content type: JSON responses; `multipart/form-data` for image upload.
- Versioning: functional endpoints are namespaced under `/v1`. Breaking changes
  ship under a new prefix (`/v2`), keeping `/v1` until deprecation.

## Authentication

Optional bearer-token auth is enabled by setting `VIT_API_TOKEN`. When set, the
`/v1/predict` endpoint requires:

```
Authorization: Bearer <token>
```

Tokens are compared in constant time. Health and metadata endpoints are
unauthenticated so orchestrators and dashboards can probe them.

## Rate limiting

`/v1/predict` is rate-limited per client IP using a 60-second sliding window
(`VIT_RATE_LIMIT_RPM`, `0` disables). Exceeding the limit returns `429` with a
`Retry-After` header. For multi-replica deployments, enforce the authoritative
limit at the ingress/gateway (the in-process limiter is per-replica).

## Endpoints

### `GET /healthz` — liveness

Always `200` while the process is running. Does **not** require a loaded model.

```json
{ "status": "ok", "model_loaded": true, "version": "1.0.0" }
```

### `GET /readyz` — readiness

`200` when a model is loaded and serveable; `503` otherwise (with the load error
in `detail`). Wire this to your readiness probe.

```json
// 503 example
{ "error": "service_unavailable", "detail": "No VIT_MODEL_CHECKPOINT configured.", "request_id": "..." }
```

### `GET /v1/metadata` — model metadata

```json
{
  "version": "1.0.0",
  "model_loaded": true,
  "num_classes": 10,
  "class_names": ["airplane", "automobile", "..."],
  "image_size": 32,
  "device": "cuda"
}
```

### `POST /v1/predict` — classify an image

**Request** — `multipart/form-data`:

| Field | Type | Notes |
|-------|------|-------|
| `file` | file (required) | JPEG/PNG/WebP/BMP, ≤ `VIT_MAX_UPLOAD_BYTES` (default 5 MiB). |
| `top_k` | query int (1–100) | Number of ranked predictions (default `VIT_DEFAULT_TOP_K`). |

**Response** — `200`:

```json
{
  "request_id": "8f14e45fceea167a5a36dedd4bea2543",
  "model_version": "1.0.0",
  "top_k": 3,
  "predictions": [
    { "label": "cat", "index": 3, "probability": 0.87 },
    { "label": "dog", "index": 5, "probability": 0.08 },
    { "label": "deer", "index": 4, "probability": 0.02 }
  ],
  "latency_ms": 12.4
}
```

**Examples**

```bash
# curl
curl -F "file=@cat.png" "http://localhost:8080/v1/predict?top_k=3" \
     -H "Authorization: Bearer $VIT_API_TOKEN"
```

```python
# Python (requests)
import requests
with open("cat.png", "rb") as f:
    r = requests.post(
        "http://localhost:8080/v1/predict",
        params={"top_k": 3},
        files={"file": ("cat.png", f, "image/png")},
        headers={"Authorization": "Bearer " + token},
    )
r.raise_for_status()
print(r.json()["predictions"])
```

### `GET /metrics` — Prometheus

Present only when `VIT_ENABLE_METRICS=true` (default). Exposes:

| Metric | Type | Labels |
|--------|------|--------|
| `vit_requests_total` | counter | method, path, status |
| `vit_request_latency_seconds` | histogram | path |
| `vit_inference_latency_seconds` | histogram | — |
| `vit_predictions_total` | counter | — |
| `vit_model_loaded` | gauge | — |

## Error model

All handled errors return a uniform envelope:

```json
{ "error": "bad_request", "detail": "Could not decode the uploaded image.", "request_id": "..." }
```

| Status | `error` code | When |
|--------|--------------|------|
| 400 | `bad_request` | Empty upload or undecodable image. |
| 401 | `unauthorized` | Missing/invalid bearer token. |
| 413 | `payload_too_large` | Upload exceeds the size limit. |
| 415 | `unsupported_media_type` | Non-image content type. |
| 422 | (FastAPI validation) | Malformed query/body (e.g. `top_k` out of range). |
| 429 | `rate_limited` | Per-IP rate limit exceeded (see `Retry-After`). |
| 503 | `service_unavailable` | Model not loaded. |
| 500 | `internal_error` | Unhandled server error (details logged, not leaked). |

Every response carries an `X-Request-ID` header (also echoed in the body and
server logs) for correlation.

## Security headers

Every response includes: `X-Content-Type-Options: nosniff`,
`X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, a restrictive
`Content-Security-Policy`, `Cross-Origin-Opener-Policy`, and `Permissions-Policy`.
See [security.md](security.md).

## Generating a static OpenAPI file

```python
import json
from vit.serving import create_app
schema = create_app().openapi()
open("openapi.json", "w").write(json.dumps(schema, indent=2))
```
