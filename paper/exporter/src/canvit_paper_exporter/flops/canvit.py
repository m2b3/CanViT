"""CanViT-B FLOPs as the paper counts them: canvit_pytorch.flops for one glimpse, plus the ADE20K probe or the ImageNet-1k
readout."""

from canvit_pytorch import CanViTConfig
from canvit_pytorch.benchmarks.ade20k import NUM_CLASSES as ADE20K_NUM_CLASSES
from canvit_pytorch.benchmarks.imagenet import NUM_CLASSES as IMAGENET_NUM_CLASSES
from canvit_pytorch.flops import (
    canvas_attention_read_flops,
    canvas_attention_write_flops,
    glimpse_flops,
    linear_flops,
    segmentation_probe_flops,
)
from canvit_pytorch.hub.repos import RELEASED_GLIMPSE_SIZE_PX

CANVIT_B = CanViTConfig()
CANVIT_B_NAME = "CanViT-B"


def segmentation_glimpse_flops(
    config: CanViTConfig, *, canvas_grid_size: int, glimpse_size_px: int = RELEASED_GLIMPSE_SIZE_PX,
) -> int:
    """One glimpse, then the ADE20K probe on the canvas: the per-glimpse cost behind every CanViT FLOP count."""
    return glimpse_flops(config, glimpse_size_px=glimpse_size_px, canvas_grid_size=canvas_grid_size) + (
        segmentation_probe_flops(grid_size=canvas_grid_size, embed_dim=config.canvas_dim, num_classes=ADE20K_NUM_CLASSES)
    )


def classification_glimpse_flops(
    config: CanViTConfig, *, canvas_grid_size: int, glimpse_size_px: int = RELEASED_GLIMPSE_SIZE_PX,
) -> int:
    """One glimpse, then the ImageNet-1k readout of the recurrent CLS token: a LayerNorm, uncounted, and a linear map."""
    return glimpse_flops(config, glimpse_size_px=glimpse_size_px, canvas_grid_size=canvas_grid_size) + linear_flops(
        num_tokens=1, in_dim=config.backbone_spec.embed_dim, out_dim=IMAGENET_NUM_CLASSES,
    )


def read_write_pair_flops(config: CanViTConfig, *, glimpse_size_px: int, canvas_grid_size: int) -> int:
    return canvas_attention_read_flops(config, glimpse_size_px=glimpse_size_px, canvas_grid_size=canvas_grid_size) + (
        canvas_attention_write_flops(config, glimpse_size_px=glimpse_size_px, canvas_grid_size=canvas_grid_size)
    )
