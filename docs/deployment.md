# Deployment Guide

The service is a stateless FastAPI app that loads a read-only checkpoint at
startup. This guide covers Docker, docker-compose, and Kubernetes.

## Configuration (environment variables)

All runtime config comes from `VIT_`-prefixed env vars (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `VIT_MODEL_CHECKPOINT` | — | Path to `best.pt`. Empty → not-ready. |
| `VIT_DEVICE` | `auto` | `auto`/`cpu`/`cuda[:N]`/`mps`. |
| `VIT_NUM_THREADS` | 0 | CPU intra-op threads (pin in containers). |
| `VIT_HOST` / `VIT_PORT` | `0.0.0.0` / 8080 | Bind address. |
| `VIT_LOG_LEVEL` / `VIT_LOG_JSON` | INFO / true | Logging. |
| `VIT_CORS_ORIGINS` | — | Comma-separated allowed origins. |
| `VIT_MAX_UPLOAD_BYTES` | 5 MiB | Upload cap. |
| `VIT_RATE_LIMIT_RPM` | 120 | Per-IP limit (0 = off). |
| `VIT_API_TOKEN` | — | Bearer token (empty = auth off). |
| `VIT_ENABLE_METRICS` | true | Expose `/metrics`. |

## Docker

```bash
# Build the runtime image (CPU wheels, non-root, healthcheck baked in)
docker build -t build-your-own-vit:latest .

# Run, mounting a checkpoint
docker run --rm -p 8080:8080 \
  -e VIT_MODEL_CHECKPOINT=/models/best.pt \
  -e VIT_DEVICE=cpu \
  -v "$(pwd)/outputs/vit_tiny_cifar10:/models:ro" \
  build-your-own-vit:latest
```

The image runs as an unprivileged user, has a read-only-friendly layout, and a
`HEALTHCHECK` hitting `/healthz`. For **GPU serving**, base the image on an
`nvidia/cuda` runtime and install a CUDA build of torch.

## docker-compose

```bash
# Brings up the API (:8080) and a Prometheus scraper (:9090)
VIT_MODELS_DIR=./outputs/vit_tiny_cifar10 docker compose up --build
```

## Kubernetes

Manifests are in [`deploy/k8s/deployment.yaml`](../deploy/k8s/deployment.yaml):
`ConfigMap`, `Deployment` (3 replicas, rolling update, hardened
`securityContext`, liveness/readiness probes, resource requests/limits),
`Service`, `HorizontalPodAutoscaler`, and a `PodDisruptionBudget`.

```bash
# 1. Provide the checkpoint via a PVC named vit-models mounted at /models
# 2. (optional) create the bearer-token secret
kubectl create secret generic vit-secrets --from-literal=api-token="$(openssl rand -hex 32)"
# 3. Point the image at your registry, then:
kubectl apply -f deploy/k8s/deployment.yaml
kubectl rollout status deployment/vit-api
```

### Autoscaling

The HPA targets 70% CPU across 3–20 replicas with a 5-minute scale-down
stabilization window. To scale on latency instead, expose
`vit_inference_latency_seconds` through the Prometheus Adapter and switch the HPA
to that custom metric.

### Rollout & rollback

Rolling updates use `maxUnavailable: 0, maxSurge: 1`, so capacity never dips
during a deploy, and the readiness gate ensures new pods take traffic only once
the model is loaded.

```bash
kubectl rollout undo deployment/vit-api          # roll back one revision
kubectl rollout undo deployment/vit-api --to-revision=3
```

Because pods are stateless and the checkpoint is immutable, rollback is
instantaneous and safe.

## Serving ONNX (optional)

```bash
vit export --checkpoint outputs/vit_tiny_cifar10/best.pt --output model.onnx
```

Serve `model.onnx` under ONNX Runtime or TensorRT for a PyTorch-free, often
lower-latency runtime. The exported graph has a dynamic batch axis.

## Production checklist

- [ ] `VIT_API_TOKEN` set (or auth handled at the gateway).
- [ ] TLS terminated at ingress; HSTS enabled.
- [ ] Rate limiting at the gateway for multi-replica correctness.
- [ ] Resource requests/limits tuned to your model size.
- [ ] Checkpoints backed up to versioned object storage.
- [ ] Prometheus scraping `/metrics`; alerts wired (see [operations.md](operations.md)).
- [ ] Image scanned (Trivy/Grype) and pinned by digest.
