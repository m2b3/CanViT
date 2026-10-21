"""Splitting a token axis into attention heads and merging the heads back."""

import torch
from torch import Tensor


def to_multihead(x: Tensor, num_heads: int) -> Tensor:
    """[B, N, D] -> [B, heads, N, D / heads]."""
    B, N, D = x.shape
    return x.view(B, N, num_heads, D // num_heads).transpose(1, 2)


def from_multihead(x: Tensor) -> Tensor:
    """[B, heads, N, head_dim] -> [B, N, heads * head_dim].

    The transposed heads are copied explicitly before the view, the copy `reshape` makes anyway.
    TorchTPU 0.2.0's `torch.compile` backend fails on CanViT-B at `transpose(1, 2).reshape(...)`
    and at `.contiguous().reshape(...)` with an invalid view; this form compiles and matches eager.
    """
    B, H, N, head_dim = x.shape
    return x.transpose(1, 2).clone(memory_format=torch.contiguous_format).view(B, N, H * head_dim)
