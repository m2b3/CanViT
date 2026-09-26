"""The ViT that processes each glimpse: DINOv3-style blocks with 2D RoPE and LayerScale."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from canvit_pytorch.model.rope import RoPE, apply_2d_rope


@dataclass(frozen=True)
class ViTSpec:
    embed_dim: int
    num_heads: int
    num_blocks: int
    patch_size: int = 16
    ffn_ratio: float = 4.0
    rope_base: float = 100.0
    layerscale_init: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def mlp_hidden_dim(self) -> int:
        return int(self.embed_dim * self.ffn_ratio)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        """[B, 3, H, W] -> [B, (H/p)(W/p), D] in row-major order."""
        return self.proj(x).flatten(2).transpose(1, 2)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SelfAttention(nn.Module):
    _bias_mask: Tensor

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, (dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        # DINOv3 has no key bias. Masking the key slice of the fused bias keeps
        # the parameters those of a plain fused Linear.
        mask = torch.ones(dim * 3)
        mask[dim : 2 * dim] = 0
        self.register_buffer("_bias_mask", mask, persistent=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, rope: RoPE) -> Tensor:
        B, N, D = x.shape
        assert self.qkv.bias is not None
        qkv = F.linear(x, self.qkv.weight, self.qkv.bias * self._bias_mask).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]  # [B, heads, N, head_dim]
        out = F.scaled_dot_product_attention(apply_2d_rope(q, rope), apply_2d_rope(k, rope), v, scale=self.head_dim**-0.5)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class ViTBlock(nn.Module):
    def __init__(self, spec: ViTSpec) -> None:
        super().__init__()
        dim = spec.embed_dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, spec.num_heads)
        self.ls1 = LayerScale(dim, spec.layerscale_init)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, spec.mlp_hidden_dim)
        self.ls2 = LayerScale(dim, spec.layerscale_init)

    def forward(self, x: Tensor, rope: RoPE) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


class ViTBackbone(nn.Module):
    """Patch embedding and transformer blocks; CanViT runs the blocks itself, interleaving Canvas Attention."""

    def __init__(self, spec: ViTSpec) -> None:
        super().__init__()
        self.spec = spec
        self.patch_embed = PatchEmbed(spec.patch_size, spec.embed_dim)
        self.blocks = nn.ModuleList([ViTBlock(spec) for _ in range(spec.num_blocks)])
        self._init_weights()

    def _init_weights(self) -> None:
        """DINOv3's initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, LayerScale):
                nn.init.constant_(module.gamma, self.spec.layerscale_init)
            elif isinstance(module, PatchEmbed):
                conv = module.proj
                bound = math.sqrt(1.0 / (conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]))
                nn.init.uniform_(conv.weight, -bound, bound)
                assert conv.bias is not None
                nn.init.uniform_(conv.bias, -bound, bound)
