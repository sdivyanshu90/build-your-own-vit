# Training Guide

How to configure, run, tune, and reproduce training.

## Running

```bash
# Smoke run: synthetic data, a few steps, no download (great for CI / a laptop)
vit train -c configs/vit_tiny_cifar10.yaml --smoke

# Full run
vit train -c configs/vit_tiny_cifar10.yaml

# Resume
vit train -c configs/vit_tiny_cifar10.yaml --resume outputs/vit_tiny_cifar10/last.pt

# Ad-hoc overrides (no file edit)
vit train -c configs/vit_tiny_cifar10.yaml --set train.epochs=50 optim.lr=5e-4
```

Outputs are written to `{train.output_dir}/{run_name}/`:

- `best.pt` — best checkpoint by `train.checkpoint_metric`.
- `last.pt` — most recent (use to resume).

## Configuration reference

A config is a validated `Config` with five sections. Unknown keys are rejected.

### `model`

| Field | Default | Meaning |
|-------|---------|---------|
| `preset` | `null` | `vit_tiny`/`small`/`base`/`large`; fills `embed_dim`, `depth`, `num_heads`, `mlp_ratio` (explicit values still win). |
| `image_size` | 224 | Input side length (must be divisible by `patch_size`). |
| `patch_size` | 16 | Square patch side. |
| `in_channels` | 3 | Input channels. |
| `num_classes` | 1000 | Output classes (must match dataset for CIFAR). |
| `embed_dim` | 192 | Token width (must be divisible by `num_heads`). |
| `depth` | 12 | Encoder blocks. |
| `num_heads` | 3 | Attention heads. |
| `mlp_ratio` | 4.0 | MLP hidden width multiplier. |
| `qkv_bias` | true | Bias in QKV projection. |
| `drop_rate` | 0.0 | Projection/MLP dropout. |
| `attn_drop_rate` | 0.0 | Attention-matrix dropout. |
| `drop_path_rate` | 0.0 | Max stochastic-depth rate (ramped by depth). |
| `pool` | `cls` | `cls` token or `mean` over patches. |
| `representation_size` | `null` | Optional pre-logits Tanh projection width. |
| `layer_norm_eps` | 1e-6 | LayerNorm epsilon. |
| `init_std` | 0.02 | Truncated-normal init std. |

### `data`

| Field | Default | Meaning |
|-------|---------|---------|
| `dataset` | `cifar10` | `cifar10`/`cifar100`/`imagefolder`/`fake`. |
| `data_dir` | `./data` | Dataset root. |
| `image_size` | 32 | Must equal `model.image_size`. |
| `batch_size` | 128 | Per-step batch. |
| `num_workers` | 4 | DataLoader workers. |
| `val_split` | 0.1 | Train fraction held out for validation when no val split exists. |
| `download` | true | Download dataset if missing. |
| `mean` / `std` | CIFAR stats | Normalization; also stored in the checkpoint for inference. |
| `random_crop_padding` | 4 | Reflect padding before random crop. |
| `horizontal_flip` | true | Random horizontal flip. |
| `rand_augment` | false | Enable RandAugment (`num_ops`, `magnitude`). |
| `mixup_alpha` / `cutmix_alpha` | 0.0 | Batch mixing strengths (soft targets). |

### `optim`

| Field | Default | Meaning |
|-------|---------|---------|
| `name` | `adamw` | `adamw` or `sgd`. |
| `lr` | 1e-3 | Peak learning rate. |
| `weight_decay` | 0.05 | Applied to matrices only (never to norms/biases/pos-embed/CLS). |
| `betas`, `eps` | (0.9,0.999), 1e-8 | AdamW params. |
| `momentum`, `nesterov` | 0.9, true | SGD params. |

### `scheduler`

| Field | Default | Meaning |
|-------|---------|---------|
| `name` | `cosine` | `cosine` (warmup + cosine decay) or `none`. |
| `warmup_epochs` | 5 | Linear warmup length (fractional allowed). |
| `min_lr` | 1e-5 | Cosine floor. |
| `warmup_start_lr` | 1e-6 | LR at step 0. |

### `train`

| Field | Default | Meaning |
|-------|---------|---------|
| `epochs` | 100 | Total epochs. |
| `precision` | `fp16` | `fp32`/`fp16`/`bf16` (autocast; fp16 uses a grad scaler on CUDA). |
| `grad_clip_norm` | 1.0 | Max grad norm (0 disables). |
| `accumulate_steps` | 1 | Micro-batches per optimizer step. |
| `label_smoothing` | 0.1 | For hard-label cross-entropy and soft targets. |
| `use_ema` / `ema_decay` | false / 0.9999 | EMA weights for eval/serving. |
| `log_interval` | 50 | Steps between train logs. |
| `output_dir` | `./outputs` | Checkpoint root. |
| `seed` / `deterministic` | 42 / false | Reproducibility. |
| `compile_model` | false | Wrap with `torch.compile`. |
| `device` | `auto` | `auto`/`cpu`/`cuda[:N]`/`mps`. |
| `num_threads` | 0 | CPU intra-op threads (0 = default). |
| `checkpoint_metric` / `checkpoint_mode` | `val_acc1` / `max` | Best-checkpoint selection. |
| `early_stopping_patience` | 0 | Epochs without improvement before stopping (0 = off). |

## Recipes & expected results

| Config | Hardware | Approx. result |
|--------|----------|----------------|
| `vit_tiny_cifar10` | 1× modern GPU, ~100 epochs | ~90%+ top-1 |
| `vit_small_cifar100` | 1× GPU, ~150 epochs | strong CIFAR-100 baseline |
| `vit_base_imagenet` | multi-GPU, 300 epochs | ImageNet-scale (adjust batch/accum) |

> ViTs need data and regularization. On small datasets, augmentation (RandAugment,
> Mixup/CutMix), stochastic depth, and EMA matter more than raw model size.

## Tuning tips

- **Warmup is not optional** for ViTs — start with 5–20 epochs of warmup.
- Scale `lr` with the *effective* batch (`batch_size × accumulate_steps × world_size`).
- If loss diverges early: lower `lr`, raise `warmup_epochs`, keep `grad_clip_norm=1.0`.
- If overfitting: raise `drop_path_rate`, enable RandAugment/Mixup/CutMix.
- Prefer `bf16` on Ampere+ (no grad scaler, more stable than fp16).

## Reproducibility

Set `train.seed` and `train.deterministic: true`. This seeds Python/NumPy/PyTorch,
sets cuDNN deterministic mode, and configures per-worker seeds. Note determinism
can reduce throughput and a few ops lack deterministic kernels (handled with
`warn_only=True`).

## Distributed training (DDP)

The loop is DDP-ready. Sketch:

```python
# torchrun --nproc_per_node=8 train_ddp.py
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# init_process_group; build model; model = DDP(model, device_ids=[local_rank])
# use DistributedSampler; guard rank-0 logging + checkpointing
```

The no-decay grouping, per-step cosine schedule, AMP, and EMA are all
DDP-compatible; only sampler wiring and rank-0 guards need adding.
