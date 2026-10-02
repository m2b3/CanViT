import math

import mlx.core as mx
from canvit_core.backbone import ViTSpec
from mlx import nn

from canvit_mlx.model.rope import RoPE, apply_2d_rope


class PatchEmbed(nn.Module):
    def __init__(self, spec: ViTSpec) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, spec.embed_dim, kernel_size=spec.patch_size, stride=spec.patch_size)
        bound = math.sqrt(1 / (3 * spec.patch_size**2))
        self.proj.bias = mx.random.uniform(-bound, bound, self.proj.bias.shape)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        return x.reshape(x.shape[0], -1, x.shape[-1])


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float) -> None:
        super().__init__()
        self.gamma = mx.full((dim,), init_value)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, rope: RoPE) -> mx.array:
        batch_size, tokens, dim = x.shape
        # DINOv3 fixes the key bias at zero, including during optimization.
        bias_mask = mx.concatenate([mx.ones(dim), mx.zeros(dim), mx.ones(dim)])
        qkv = x @ self.qkv.weight.T + self.qkv.bias * bias_mask
        qkv = qkv.reshape(batch_size, tokens, 3, self.num_heads, dim // self.num_heads)
        query, key, value = (qkv[:, :, index].transpose(0, 2, 1, 3) for index in range(3))
        output = mx.fast.scaled_dot_product_attention(
            apply_2d_rope(query, rope),
            apply_2d_rope(key, rope),
            value,
            scale=(dim // self.num_heads) ** -0.5,
        )
        return self.proj(output.transpose(0, 2, 1, 3).reshape(batch_size, tokens, dim))


class ViTBlock(nn.Module):
    def __init__(self, spec: ViTSpec) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(spec.embed_dim)
        self.attn = SelfAttention(spec.embed_dim, spec.num_heads)
        self.ls1 = LayerScale(spec.embed_dim, spec.layerscale_init)
        self.norm2 = nn.LayerNorm(spec.embed_dim)
        self.mlp = MLP(spec.embed_dim, spec.mlp_hidden_dim)
        self.ls2 = LayerScale(spec.embed_dim, spec.layerscale_init)

    def __call__(self, x: mx.array, rope: RoPE) -> mx.array:
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


class ViTBackbone(nn.Module):
    def __init__(self, spec: ViTSpec) -> None:
        super().__init__()
        self.spec = spec
        self.patch_embed = PatchEmbed(spec)
        self.blocks = [ViTBlock(spec) for _ in range(spec.num_blocks)]
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.weight = mx.random.truncated_normal(-100, 100, shape=module.weight.shape) * 0.02
                module.bias = mx.zeros_like(module.bias)
