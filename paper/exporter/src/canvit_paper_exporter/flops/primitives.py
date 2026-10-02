"""FLOP formulas for the prior active models (AME, AdaGlimpse). One MAC = two FLOPs (PyTorch flop_counter convention)."""

# MLP hidden size over embedding size in AME's and AdaGlimpse's ViT blocks.
FFN_RATIO: float = 4.0


def linear(n_tokens: int, d_in: int, d_out: int) -> int:
    return 2 * n_tokens * d_in * d_out


def sdpa(n_q: int, n_kv: int, d: int) -> int:
    """`d` is total embedding dim, summed across heads."""
    return 4 * n_q * n_kv * d


def conv2d(h: int, w: int, c_in: int, c_out: int, k: int) -> int:
    """Square-kernel 2D convolution."""
    return 2 * h * w * c_in * c_out * k * k


def patch_embed(n_patches: int, patch_size: int, dim: int) -> int:
    """Conv2d(3 -> dim, kernel=patch_size) producing `n_patches` tokens."""
    return 2 * n_patches * patch_size * patch_size * 3 * dim


def vit_block(n: int, dim: int, ffn_ratio: float) -> int:
    ffn = int(ffn_ratio * dim)
    return (
        linear(n, dim, 3 * dim)
        + sdpa(n, n, dim)
        + linear(n, dim, dim)
        + linear(n, dim, ffn)
        + linear(n, ffn, dim)
    )
