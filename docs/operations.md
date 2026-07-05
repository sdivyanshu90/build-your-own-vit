# Operational Runbook

Day-2 operations: what to watch, what alerts to set, and how to respond.

## Observability

### Logs

Structured JSON (set `VIT_LOG_JSON=true`), one object per line, with
`request_id` correlation. Ship to Loki/CloudWatch/Stackdriver. Key events:
`Model loaded`, `Model load failed`, `train step`, `Epoch complete`,
`Unhandled server error`.

### Metrics (Prometheus, `/metrics`)

| Metric | Use |
|--------|-----|
| `vit_model_loaded` | Readiness gauge (1/0). |
| `vit_requests_total{status}` | Traffic & error-rate (5xx/total). |
| `vit_request_latency_seconds` | End-to-end latency histogram (p50/p95/p99). |
| `vit_inference_latency_seconds` | Model-only latency. |
| `vit_predictions_total` | Successful predictions throughput. |

### Suggested alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| ServiceDown | `up == 0` for 2m | page |
| NotReady | `vit_model_loaded == 0` for 5m | page |
| HighErrorRate | 5xx ratio > 5% for 5m | page |
| HighLatency | p99 `vit_request_latency_seconds` > 1s for 10m | ticket |
| SaturationCPU | pod CPU > 85% for 10m (HPA should react) | ticket |

## Health probes

- **Liveness** `/healthz` → restart the container if the process is wedged.
- **Readiness** `/readyz` → withhold traffic until the model loads; also flips to
  503 if the model becomes unavailable.

## Common operations

### Deploy a new model

1. Train and validate; copy `best.pt` to the models store (versioned).
2. Update `VIT_MODEL_CHECKPOINT` (or the PVC contents) and roll the deployment.
3. Watch `vit_model_loaded` → 1 and error rate steady.

### Roll back

```bash
kubectl rollout undo deployment/vit-api
```

Stateless pods + immutable checkpoint make this instant and safe.

### Scale

HPA auto-scales on CPU (3–20 replicas). Manual override:

```bash
kubectl scale deployment/vit-api --replicas=8
```

### Rotate the API token

Update the `vit-secrets` secret and `kubectl rollout restart deployment/vit-api`.

## Incident response

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `/readyz` 503 across pods | Bad/missing checkpoint path | Check `Model load failed` logs; fix `VIT_MODEL_CHECKPOINT`; redeploy. |
| Rising 5xx | Bad input handling / resource exhaustion | Inspect `request_id` logs; check memory/CPU; scale out. |
| Latency spike | Under-provisioned / cold pods | Verify HPA scaled; increase requests/limits; pin `VIT_NUM_THREADS`. |
| 429s | Rate limit too low / abuse | Raise `VIT_RATE_LIMIT_RPM` or handle at gateway; investigate source IPs. |
| OOMKilled | Model too big for limits | Raise memory limit; use a smaller preset or ONNX runtime. |

## Capacity planning

- CPU inference latency scales with model size and `image_size`; benchmark with a
  representative load and set requests to p95 CPU.
- Pin `VIT_NUM_THREADS` to the pod's CPU limit so PyTorch doesn't oversubscribe
  the host's core count inside a cgroup.
- For high QPS, prefer more small replicas over few large ones (better tail
  latency and rolling-update headroom).

## Backup & DR

The only stateful artifact is the checkpoint. Back it up to a versioned bucket;
recovery is redeploying pointed at the last-good object. Target RPO = last saved
checkpoint; RTO = pod start + model load (seconds to a minute).
