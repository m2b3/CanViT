from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ViTSpec:
    embed_dim: int
    num_heads: int
    num_blocks: int
    patch_size: int = 16
    ffn_ratio: float = 4.0
    rope_base: float = 100.0
    layerscale_init: float = 1e-5

    def __post_init__(self) -> None:
        assert all(
            type(value) is int for value in (self.embed_dim, self.num_heads, self.num_blocks, self.patch_size)
        ), self
        assert min(self.embed_dim, self.num_heads, self.num_blocks, self.patch_size) > 0, self
        assert self.embed_dim % self.num_heads == 0 and self.head_dim % 4 == 0, self
        assert self.mlp_hidden_dim > 0 and self.rope_base > 0, self

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def mlp_hidden_dim(self) -> int:
        return int(self.embed_dim * self.ffn_ratio)


BackboneName = Literal["vits16", "vitb16"]

BACKBONES: dict[BackboneName, ViTSpec] = {
    "vits16": ViTSpec(embed_dim=384, num_heads=6, num_blocks=12),
    "vitb16": ViTSpec(embed_dim=768, num_heads=12, num_blocks=12),
}
