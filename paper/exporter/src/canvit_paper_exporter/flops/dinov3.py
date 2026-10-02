"""The DINOv3 ViTs CanViT is compared with, for FLOP counts.

The facebook/dinov3-* repos are gated on the Hub, so their architectures are stated
here; canvit_pytorch.flops.dinov3_flops counts from them.
"""

from transformers import DINOv3ViTConfig

DINOV3_PATCH_SIZE = 16

DINOV3_CONFIGS: dict[str, DINOv3ViTConfig] = {
    "DINOv3 ViT-S/16": DINOv3ViTConfig(
        hidden_size=384, intermediate_size=1536, num_hidden_layers=12, num_attention_heads=6,
        num_register_tokens=4, patch_size=DINOV3_PATCH_SIZE,
    ),
    "DINOv3 ViT-B/16": DINOv3ViTConfig(
        hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12,
        num_register_tokens=4, patch_size=DINOV3_PATCH_SIZE,
    ),
}
