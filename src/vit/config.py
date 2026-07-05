"""Typed, validated configuration for models, data, and training.

Configuration is expressed with Pydantic v2 models. This gives us:

* **Validation at the boundary** — an invalid ``patch_size`` fails immediately
  with a precise message instead of a cryptic shape error deep in the forward
  pass.
* **Self-documentation** — every field has a description surfaced in JSON Schema
  (``Config.model_json_schema()``), usable by editors and docs tooling.
* **One source of truth** — YAML files, CLI overrides, and programmatic use all
  funnel through the same models.

The composition is a single :class:`Config` aggregate of section models, mirrored
by the YAML files under ``configs/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Architecture presets (name -> partial ModelConfig overrides).
# Sizes follow Dosovitskiy et al. (2021) and the DeiT paper conventions.
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict[str, int | float]] = {
    "vit_tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3, "mlp_ratio": 4.0},
    "vit_small": {"embed_dim": 384, "depth": 12, "num_heads": 6, "mlp_ratio": 4.0},
    "vit_base": {"embed_dim": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0},
    "vit_large": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4.0},
}


class _Base(BaseModel):
    """Shared model config: reject unknown keys to catch typos in YAML."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelConfig(_Base):
    """Vision Transformer architecture hyper-parameters."""

    preset: str | None = Field(
        default=None,
        description="Named preset (vit_tiny/small/base/large). Sets embed_dim, "
        "depth, num_heads, mlp_ratio unless explicitly overridden.",
    )
    image_size: PositiveInt = Field(default=224, description="Input side length.")
    patch_size: PositiveInt = Field(default=16, description="Square patch side.")
    in_channels: PositiveInt = Field(default=3, description="Input channels.")
    num_classes: PositiveInt = Field(default=1000, description="Output classes.")
    embed_dim: PositiveInt = Field(default=192, description="Token embedding size.")
    depth: PositiveInt = Field(default=12, description="Number of encoder blocks.")
    num_heads: PositiveInt = Field(default=3, description="Attention heads.")
    mlp_ratio: PositiveFloat = Field(default=4.0, description="MLP hidden ratio.")
    qkv_bias: bool = Field(default=True, description="Learn a bias for Q/K/V.")
    drop_rate: NonNegativeFloat = Field(default=0.0, description="Projection dropout.")
    attn_drop_rate: NonNegativeFloat = Field(default=0.0, description="Attention-matrix dropout.")
    drop_path_rate: NonNegativeFloat = Field(
        default=0.0, description="Max stochastic-depth drop probability."
    )
    pool: Literal["cls", "mean"] = Field(
        default="cls", description="Head pooling: class token or mean over patches."
    )
    representation_size: PositiveInt | None = Field(
        default=None, description="Optional pre-logits projection width."
    )
    layer_norm_eps: PositiveFloat = Field(default=1e-6, description="LayerNorm eps.")
    init_std: PositiveFloat = Field(
        default=0.02, description="Std for truncated-normal weight init."
    )

    @model_validator(mode="after")
    def _apply_preset_and_validate(self) -> ModelConfig:
        if self.preset is not None:
            if self.preset not in PRESETS:
                raise ValueError(f"Unknown preset {self.preset!r}. Available: {sorted(PRESETS)}")
            # Only fill fields the user left at their default so explicit
            # overrides in YAML always win over the preset.
            explicit = self.model_fields_set
            for key, value in PRESETS[self.preset].items():
                if key not in explicit:
                    object.__setattr__(self, key, value)

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})."
            )
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        if self.drop_path_rate >= 1.0 or self.drop_rate >= 1.0:
            raise ValueError("Dropout / drop-path probabilities must be < 1.0.")
        return self

    @property
    def num_patches(self) -> int:
        """Number of patch tokens (excluding the class token)."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def seq_len(self) -> int:
        """Full sequence length fed to the encoder (patches + class token)."""
        return self.num_patches + (1 if self.pool == "cls" else 0)

    @classmethod
    def from_preset(cls, preset: str, **overrides: Any) -> ModelConfig:
        """Construct a config from a preset name with optional overrides."""
        return cls(preset=preset, **overrides)


class DataConfig(_Base):
    """Dataset and data-loading configuration."""

    dataset: Literal["cifar10", "cifar100", "imagefolder", "fake"] = Field(
        default="cifar10", description="Dataset backend."
    )
    data_dir: str = Field(default="./data", description="Dataset root directory.")
    image_size: PositiveInt = Field(default=32, description="Resized image side.")
    batch_size: PositiveInt = Field(default=128, description="Per-step batch size.")
    num_workers: NonNegativeInt = Field(default=4, description="DataLoader workers.")
    pin_memory: bool = Field(default=True, description="Pin host memory for H2D copies.")
    val_split: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description="Fraction of train used for val "
        "when a dataset has no dedicated validation split.",
    )
    download: bool = Field(default=True, description="Download dataset if missing.")
    # Normalization statistics (defaults are CIFAR's; overridable for ImageNet).
    mean: tuple[float, float, float] = Field(default=(0.4914, 0.4822, 0.4465))
    std: tuple[float, float, float] = Field(default=(0.2470, 0.2435, 0.2616))
    # Augmentation toggles (train split only).
    random_crop_padding: NonNegativeInt = Field(default=4)
    horizontal_flip: bool = Field(default=True)
    rand_augment: bool = Field(default=False)
    rand_augment_num_ops: PositiveInt = Field(default=2)
    rand_augment_magnitude: NonNegativeInt = Field(default=9)
    # Batch-level regularizers (applied in the training loop).
    mixup_alpha: NonNegativeFloat = Field(default=0.0)
    cutmix_alpha: NonNegativeFloat = Field(default=0.0)

    @field_validator("mean", "std", mode="before")
    @classmethod
    def _coerce_stats(cls, v: Any) -> Any:
        if isinstance(v, list | tuple) and len(v) != 3:
            raise ValueError("mean/std must have exactly 3 channel values.")
        return v


class OptimConfig(_Base):
    """Optimizer configuration."""

    name: Literal["adamw", "sgd"] = Field(default="adamw")
    lr: PositiveFloat = Field(default=1e-3, description="Peak learning rate.")
    weight_decay: NonNegativeFloat = Field(default=0.05)
    betas: tuple[float, float] = Field(default=(0.9, 0.999))
    eps: PositiveFloat = Field(default=1e-8)
    momentum: NonNegativeFloat = Field(default=0.9, description="SGD momentum.")
    nesterov: bool = Field(default=True, description="SGD Nesterov momentum.")


class SchedulerConfig(_Base):
    """Learning-rate schedule configuration."""

    name: Literal["cosine", "none"] = Field(default="cosine")
    warmup_epochs: NonNegativeFloat = Field(default=5.0)
    min_lr: NonNegativeFloat = Field(default=1e-5)
    warmup_start_lr: NonNegativeFloat = Field(default=1e-6)


class TrainConfig(_Base):
    """Training-loop configuration."""

    epochs: PositiveInt = Field(default=100)
    precision: Literal["fp32", "fp16", "bf16"] = Field(default="fp16")
    grad_clip_norm: NonNegativeFloat = Field(
        default=1.0, description="Max grad norm (0 disables clipping)."
    )
    accumulate_steps: PositiveInt = Field(default=1)
    label_smoothing: float = Field(default=0.1, ge=0.0, lt=1.0)
    use_ema: bool = Field(default=False)
    ema_decay: float = Field(default=0.9999, gt=0.0, lt=1.0)
    log_interval: PositiveInt = Field(default=50, description="Steps between logs.")
    output_dir: str = Field(default="./outputs")
    seed: NonNegativeInt = Field(default=42)
    deterministic: bool = Field(default=False)
    compile_model: bool = Field(default=False, description="Wrap with torch.compile.")
    device: str = Field(default="auto")
    num_threads: NonNegativeInt = Field(default=0)
    checkpoint_metric: str = Field(default="val_acc1")
    checkpoint_mode: Literal["max", "min"] = Field(default="max")
    early_stopping_patience: NonNegativeInt = Field(
        default=0, description="Epochs without improvement before stopping (0=off)."
    )


class Config(_Base):
    """Top-level experiment configuration aggregating all sections."""

    run_name: str = Field(default="vit-run")
    tags: list[str] = Field(default_factory=list)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @model_validator(mode="after")
    def _cross_section_consistency(self) -> Config:
        # Keep the model's spatial/label geometry in sync with the data section.
        if self.model.image_size != self.data.image_size:
            raise ValueError(
                f"model.image_size ({self.model.image_size}) must match "
                f"data.image_size ({self.data.image_size})."
            )
        expected = _DATASET_NUM_CLASSES.get(self.data.dataset)
        if expected is not None and self.model.num_classes != expected:
            raise ValueError(
                f"model.num_classes ({self.model.num_classes}) must equal "
                f"{expected} for dataset {self.data.dataset!r}."
            )
        return self

    def to_yaml(self) -> str:
        """Serialize to a YAML string (round-trippable via :func:`load_config`)."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


_DATASET_NUM_CLASSES: dict[str, int] = {
    "cifar10": 10,
    "cifar100": 100,
    # imagefolder / fake are dynamic → not enforced here.
}


def load_config(path: str | Path, **overrides: Any) -> Config:
    """Load a :class:`Config` from a YAML file, applying dotted overrides.

    Args:
        path: Path to a YAML config file.
        **overrides: Dotted-key overrides, e.g. ``**{"train.epochs": 5}``.
            Values replace the corresponding nested field before validation.

    Returns:
        A validated :class:`Config`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the YAML is not a mapping or validation fails.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}.")
    for dotted, value in overrides.items():
        _set_dotted(raw, dotted, value)
    return Config.model_validate(raw)


def _set_dotted(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``target["a"]["b"] = value`` from the dotted key ``"a.b"``."""
    keys = dotted_key.split(".")
    node = target
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def save_config(config: Config, path: str | Path) -> Path:
    """Write a config to ``path`` as YAML, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.to_yaml(), encoding="utf-8")
    return path
