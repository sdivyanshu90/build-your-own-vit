"""Multi-head self-attention.

This is the core mixing operation of the Transformer. For a sequence of ``N``
tokens of width ``D`` split into ``H`` heads (each of width ``d = D/H``):

1. Project the input to queries, keys and values: ``QKV = x · W_qkv``.
2. For each head, compute ``softmax(Q·Kᵀ / √d) · V``.
3. Concatenate heads and project back to ``D``.

Two execution paths are provided:

* **Fused** (default): :func:`torch.nn.functional.scaled_dot_product_attention`,
  which dispatches to FlashAttention / memory-efficient kernels when available.
* **Manual**: an explicit implementation used when the caller requests the
  attention matrix (``need_weights=True``) for visualization or analysis.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with a fused fast path.

    Args:
        embed_dim: Token width ``D``. Must be divisible by ``num_heads``.
        num_heads: Number of attention heads ``H``.
        qkv_bias: Whether the QKV projection learns a bias.
        attn_drop: Dropout probability applied to the attention matrix.
        proj_drop: Dropout probability applied to the output projection.

    Raises:
        ValueError: If ``embed_dim`` is not divisible by ``num_heads``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, *, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run self-attention over ``x`` of shape ``(B, N, D)``.

        Args:
            x: Input token sequence.
            need_weights: When ``True``, also return the averaged-over-heads
                attention matrix of shape ``(B, N, N)`` and use the manual path.

        Returns:
            ``(output, weights)`` where ``output`` has shape ``(B, N, D)`` and
            ``weights`` is ``None`` unless ``need_weights`` is set.
        """
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected input of shape (B, N, {self.embed_dim}), got {tuple(x.shape)}."
            )
        batch, seq_len, _ = x.shape
        # (B, N, 3D) -> (3, B, H, N, d)
        qkv = (
            self.qkv(x)
            .reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)

        if need_weights:
            output, weights = self._manual_attention(query, key, value)
        else:
            # Fused kernel; dropout is applied inside SDPA during training.
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.attn_drop_p if self.training else 0.0,
            )
            weights = None

        # (B, H, N, d) -> (B, N, D)
        output = output.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)
        output = self.proj_drop(self.proj(output))
        return output, weights

    def _manual_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explicit attention that also returns head-averaged weights."""
        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        output = attn @ value
        # Average across heads for a single interpretable (B, N, N) map.
        weights = attn.mean(dim=1)
        return output, weights
