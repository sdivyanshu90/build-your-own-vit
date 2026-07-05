# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Placeholder for upcoming changes.

## [1.0.0] - 2026-07-05

Initial production release.

### Added
- **Model**: from-scratch Vision Transformer — `PatchEmbedding`,
  `MultiHeadSelfAttention` (fused SDPA + interpretable manual path), `MLP`,
  `DropPath` (stochastic depth), pre-norm `EncoderBlock`/`TransformerEncoder`,
  and `VisionTransformer` with CLS/mean pooling, optional pre-logits, and
  attention-map extraction. Presets: `vit_tiny`/`small`/`base`/`large` with
  verified parameter counts.
- **Config**: Pydantic v2 schema (`model`/`data`/`optim`/`scheduler`/`train`)
  with cross-section validation, presets, dotted-key overrides, and YAML I/O.
- **Data**: `DataModule` for CIFAR-10/100, ImageFolder, and a synthetic `fake`
  backend; train/eval transform pipelines (crop, flip, RandAugment, normalize).
- **Training**: `Trainer` with AMP (fp16/bf16), gradient accumulation & clipping,
  AdamW no-decay grouping, cosine+warmup schedule, Mixup/CutMix with soft
  targets, label smoothing, EMA, early stopping, and atomic self-describing
  checkpoints (best + last, resumable).
- **Inference**: `Predictor` (rebuilds model from checkpoint) and ONNX export
  with a dynamic batch axis.
- **Serving**: FastAPI app with `/healthz`, `/readyz`, `/v1/predict`,
  `/v1/metadata`, `/metrics`; security headers, per-IP rate limiting, optional
  bearer auth, upload validation, structured JSON logs, and Prometheus metrics.
- **CLI**: `vit` entry point — `train`, `evaluate`, `predict`, `export`,
  `serve`, `init-config`, `info`.
- **DevOps**: multi-stage non-root Dockerfile, docker-compose (+ Prometheus),
  Kubernetes manifests (Deployment, Service, HPA, PDB, probes), GitHub Actions CI
  (lint, type-check, test matrix, image smoke test) and release pipeline.
- **Quality**: Ruff + Mypy, Pytest suite at **96%+** coverage, pre-commit hooks.
- **Docs**: architecture, API, training, deployment, security, operations, and
  troubleshooting guides.

[Unreleased]: https://github.com/your-org/build-your-own-vit/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/your-org/build-your-own-vit/releases/tag/v1.0.0
