"""The DINOv3 ViTs CanViT is compared with, for FLOP counts.

The facebook/dinov3-* repos are gated on the Hub, so their architectures are stated
here; canvit_pytorch.flops.dinov3_flops counts from them.
"""

from canvit_pytorch.benchmarks.ade20k import NUM_CLASSES as ADE20K_NUM_CLASSES
from canvit_pytorch.flops import dinov3_flops, segmentation_probe_flops
from canvit_pytorch.teacher import DINOV3_PATCH_SIZE
from transformers import DINOv3ViTConfig

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


def dinov3_segmentation_flops(name: str, *, input_size_px: int) -> int:
    """A DINOv3 ViT on a square image, then its ADE20K probe on the output patch grid: counted as CanViT's are."""
    config = DINOV3_CONFIGS[name]
    return dinov3_flops(config, input_size_px=input_size_px) + segmentation_probe_flops(
        grid_size=input_size_px // DINOV3_PATCH_SIZE, embed_dim=config.hidden_size, num_classes=ADE20K_NUM_CLASSES)
