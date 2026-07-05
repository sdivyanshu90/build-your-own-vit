"""Transformer encoder block and stack.

Each block follows the **pre-normalization** design used by ViT/DeiT, which is
markedly more stable to train than the original post-norm Transformer:

    x = x + DropPath(Attention(LayerNorm(x)))
    x = x + DropPath(MLP(LayerNorm(x)))

The residual connections keep gradients flowing; the two LayerNorms sit *inside*
the residual branches (pre-norm) rather than after the addition (post-norm).
"""

from __future__ import annotations

import torch
from torch import nn

from vit.models.attention import MultiHeadSelfAttention
from vit.models.drop_path import DropPath
from vit.models.mlp import MLP


class EncoderBlock(nn.Module):
    """A single pre-norm Transformer encoder block.

    Args:
        embed_dim: Token width.
        num_heads: Attention heads.
        mlp_ratio: Hidden-to-embed width ratio for the MLP.
        qkv_bias: Learn a bias in the QKV projection.
        drop: Dropout for projections and the MLP.
        attn_drop: Dropout on the attention matrix.
        drop_path: Stochastic-depth probability for this block.
        layer_norm_eps: Epsilon for LayerNorm numerical stability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attn = MultiHeadSelfAttention(
            embed_dim,
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.mlp = MLP(
            embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(
        self, x: torch.Tensor, *, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the block. Returns ``(x, attn_weights_or_None)``."""
        attn_out, weights = self.attn(self.norm1(x), need_weights=need_weights)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x, weights


class TransformerEncoder(nn.Module):
    """A stack of :class:`EncoderBlock` with linearly-ramped stochastic depth.

    Args:
        depth: Number of blocks.
        embed_dim: Token width.
        num_heads: Attention heads.
        mlp_ratio: MLP hidden ratio.
        qkv_bias: Learn QKV bias.
        drop: Projection/MLP dropout.
        attn_drop: Attention dropout.
        drop_path_rate: Maximum stochastic-depth rate; ramped 0→this across depth.
        layer_norm_eps: LayerNorm epsilon.
    """

    def __init__(
        self,
        depth: int,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        # Linearly increasing drop-path probability with depth (stochastic depth).
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            EncoderBlock(
                embed_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                layer_norm_eps=layer_norm_eps,
            )
            for i in range(depth)
        )

    def forward(
        self, x: torch.Tensor, *, return_attention: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run all blocks.

        Args:
            x: Token sequence ``(B, N, D)``.
            return_attention: Collect per-layer attention maps.

        Returns:
            ``(x, attentions)`` where ``attentions`` is an empty list unless
            ``return_attention`` is set, in which case it holds one ``(B, N, N)``
            tensor per block.
        """
        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            x, weights = block(x, need_weights=return_attention)
            if return_attention and weights is not None:
                attentions.append(weights)
        return x, attentions
