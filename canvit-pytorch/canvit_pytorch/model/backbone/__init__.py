"""The ViT backbones CanViT is built on, by name."""

from typing import Literal

from canvit_pytorch.model.backbone.vit import ViTBackbone, ViTSpec

BackboneName = Literal["vits16", "vitb16"]

BACKBONES: dict[BackboneName, ViTSpec] = {
    "vits16": ViTSpec(embed_dim=384, num_heads=6, num_blocks=12),
    "vitb16": ViTSpec(embed_dim=768, num_heads=12, num_blocks=12),
}

__all__ = ["BACKBONES", "BackboneName", "ViTBackbone", "ViTSpec"]
