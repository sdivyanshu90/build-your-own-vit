"""The Vision Transformer model.

Assembles the full architecture from Dosovitskiy et al. (2021),
*"An Image Is Worth 16x16 Words"*:

    image → PatchEmbedding → prepend [CLS] → + positional embeddings
          → TransformerEncoder → LayerNorm → pooling → (pre-logits) → head

The model is configured entirely by a :class:`~vit.config.ModelConfig`, and can
round-trip to/from that config, which is what makes checkpoints self-describing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from vit.models.encoder import TransformerEncoder
from vit.models.patch_embedding import PatchEmbedding

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vit.config import ModelConfig


class VisionTransformer(nn.Module):
    """Vision Transformer image classifier.

    Prefer constructing via :meth:`from_config` (or
    :func:`vit.models.build_model`) so that validation and preset resolution are
    handled consistently.
    """

    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        pool: str = "cls",
        representation_size: int | None = None,
        layer_norm_eps: float = 1e-6,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if pool not in {"cls", "mean"}:
            raise ValueError(f"pool must be 'cls' or 'mean', got {pool!r}.")
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.pool = pool
        self.init_std = init_std

        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token (only used for cls-pooling) and positional embeddings.
        self.use_cls_token = pool == "cls"
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_tokens = num_patches + 1
        else:
            self.register_parameter("cls_token", None)
            num_tokens = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.encoder = TransformerEncoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            layer_norm_eps=layer_norm_eps,
        )
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        # Optional non-linear pre-logits projection (the "representation" layer).
        if representation_size:
            self.pre_logits: nn.Module = nn.Sequential(
                nn.Linear(embed_dim, representation_size), nn.Tanh()
            )
            head_in = representation_size
        else:
            self.pre_logits = nn.Identity()
            head_in = embed_dim

        self.head = nn.Linear(head_in, num_classes)
        self._init_weights()

    # -- initialization ----------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=self.init_std)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=self.init_std)
        self.apply(self._init_module)

    def _init_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            # Fan-in init scaled for the patch projection.
            fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            nn.init.trunc_normal_(module.weight, std=math.sqrt(1.0 / fan_in))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # -- forward -----------------------------------------------------------
    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify, prepend the class token, and add positional embeddings."""
        x = self.patch_embed(x)  # (B, N, D)
        if self.cls_token is not None:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        return cast(torch.Tensor, self.pos_drop(x))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled, normalized feature vector ``(B, D)`` before the head."""
        tokens = self._tokenize(x)
        encoded, _ = self.encoder(tokens)
        normed = self.norm(encoded)
        # CLS pooling takes the class token; mean pooling averages patch tokens.
        pooled = normed[:, 0] if self.pool == "cls" else normed.mean(dim=1)
        return cast(torch.Tensor, self.pre_logits(pooled))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images ``(B, C, H, W)`` → logits ``(B, num_classes)``."""
        return cast(torch.Tensor, self.head(self.forward_features(x)))

    @torch.no_grad()
    def get_attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return per-layer head-averaged attention maps for visualization.

        Each element has shape ``(B, N, N)``. Runs in eval mode semantics
        (no dropout randomness) under ``no_grad``.
        """
        was_training = self.training
        self.eval()
        try:
            tokens = self._tokenize(x)
            _, attentions = self.encoder(tokens, return_attention=True)
        finally:
            self.train(was_training)
        return cast("list[torch.Tensor]", attentions)

    # -- introspection / round-trip ---------------------------------------
    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: ModelConfig) -> VisionTransformer:
        """Build a model from a :class:`~vit.config.ModelConfig`."""
        return cls(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
            pool=config.pool,
            representation_size=config.representation_size,
            layer_norm_eps=config.layer_norm_eps,
            init_std=config.init_std,
        )
