# Troubleshooting

Common issues and fixes, grouped by phase.

## Installation

**`torch` install is slow or picks the wrong build.**
Install the CPU wheels explicitly, then the package:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[all]"
```

**`ImportError` for `fastapi`/`onnx`/`prometheus_client`.**
Those live in extras: `pip install -e ".[serve,export]"`.

## Configuration / validation

**`ValidationError: image_size ... must be divisible by patch_size`.**
Pick compatible values (e.g. 32/4, 224/16).

**`model.num_classes ... must equal 10 for dataset 'cifar10'`.**
CIFAR-10/100 have fixed class counts; align `model.num_classes` (10/100).

**`model.image_size ... must match data.image_size`.**
Keep them equal; the model's spatial geometry is fixed at build time.

**`--set optim.lr=1e-3` seems ignored.**
Handled: values are coerced (YAML 1.1 parses unsigned-exponent floats as
strings, so the CLI retries a float conversion). Prefer `5e-4` style or `0.0005`.

## Training

**Loss is NaN or diverges early.**
Lower `optim.lr`, increase `scheduler.warmup_epochs`, keep `grad_clip_norm=1.0`,
and prefer `bf16` over `fp16` on Ampere+.

**`fp16` training unstable / grad scaler errors.**
fp16 needs a grad scaler (enabled automatically on CUDA). On CPU, fp16 autocast
is disabled by design — use `fp32` or `bf16`.

**Out of memory (CUDA).**
Reduce `data.batch_size`; increase `train.accumulate_steps` to keep the effective
batch; try a smaller preset; enable `bf16`.

**Training is CPU-bound / slow data loading.**
Raise `data.num_workers`, ensure `pin_memory: true`, and keep persistent workers
(automatic when `num_workers > 0`).

**Non-deterministic results.**
Set `train.seed` and `train.deterministic: true`. Some ops lack deterministic
kernels (we use `warn_only=True`); expect a small throughput cost.

**Downloads fail in CI / offline.**
Use the synthetic backend: `--smoke` (or `data.dataset: fake`). No network needed.

## Inference

**`FileNotFoundError: Checkpoint not found`.**
Check the path; training writes to `{output_dir}/{run_name}/best.pt`.

**Predictions look random.**
The model may be undertrained, or you loaded raw weights when EMA is better —
try/omit `--no-ema`. Verify the checkpoint's `preprocess` matches your inputs.

**`Checkpoint format version N is newer than supported`.**
Upgrade the `vit` package; the checkpoint was written by a newer version.

## Serving

**`/readyz` returns 503.**
No/invalid `VIT_MODEL_CHECKPOINT`. Check logs for `Model load failed` and the
`detail` field; fix the path and restart.

**`415 unsupported_media_type`.**
Send an allowed image type (`image/jpeg|png|webp|bmp`) with the correct
`Content-Type` on the `file` part.

**`413 payload_too_large`.**
Image exceeds `VIT_MAX_UPLOAD_BYTES` (default 5 MiB). Downscale or raise the cap.

**`401 unauthorized`.**
`VIT_API_TOKEN` is set; send `Authorization: Bearer <token>`.

**`429 rate_limited`.**
You hit `VIT_RATE_LIMIT_RPM`. Back off (see `Retry-After`) or raise the limit.
For multiple replicas, enforce limits at the gateway.

**`/metrics` returns 404.**
Metrics are disabled (`VIT_ENABLE_METRICS=false`). Enable it.

## Docker / Kubernetes

**Container marked unhealthy.**
`HEALTHCHECK` hits `/healthz`; if failing, the process likely didn't start —
`docker logs` / `kubectl logs` for the traceback.

**Pod stuck `Running` but `0/1 ready`.**
Readiness probe (`/readyz`) failing → model not loaded. Verify the PVC mount and
`VIT_MODEL_CHECKPOINT`.

**PyTorch oversubscribes CPU in a pod.**
Set `VIT_NUM_THREADS` to the pod's CPU limit.

Still stuck? Open an issue with your config, the full JSON logs (including
`request_id`), and `vit info` output.
