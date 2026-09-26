"""Analytic forward FLOPs of CanViT and of the DINOv3 ViTs it is compared with.

One multiply-accumulate counts as two FLOPs. The counts cover what
torch.utils.flop_counter counts: matrix products (linear layers, both products
of attention) and convolutions (patch embeddings, the segmentation probe).
Normalization, softmax, RoPE, activations and residual additions are left out.
"""

from typing import TYPE_CHECKING, assert_never

from canvit_pytorch.model.canvit import canvas_attention_schedule
from canvit_pytorch.model.config import CanViTConfig

if TYPE_CHECKING:
    from transformers import DINOv3ViTConfig


def linear_flops(*, num_tokens: int, in_dim: int, out_dim: int) -> int:
    return 2 * num_tokens * in_dim * out_dim


def attention_flops(*, num_queries: int, num_keys: int, dim: int) -> int:
    """Query-key scores, then the weighted sum of values; dim is summed over heads."""
    return 4 * num_queries * num_keys * dim


def num_glimpse_tokens(config: CanViTConfig, *, glimpse_size_px: int) -> int:
    """The glimpse stream: the VPE token when enabled, the recurrent CLS token, backbone registers, glimpse patches."""
    return int(config.enable_vpe) + 1 + config.num_backbone_registers + _glimpse_grid_size(config, glimpse_size_px) ** 2


def num_canvas_tokens(config: CanViTConfig, *, canvas_grid_size: int) -> int:
    return config.num_canvas_registers + canvas_grid_size**2


def canvas_attention_read_flops(config: CanViTConfig, *, glimpse_size_px: int, canvas_grid_size: int) -> int:
    """One Canvas Attention Read: glimpse tokens query the canvas."""
    num_glimpse = num_glimpse_tokens(config, glimpse_size_px=glimpse_size_px)
    num_canvas = num_canvas_tokens(config, canvas_grid_size=canvas_grid_size)
    backbone_dim, canvas_dim = config.backbone_spec.embed_dim, config.canvas_dim
    return (
        linear_flops(num_tokens=num_glimpse, in_dim=backbone_dim, out_dim=canvas_dim)  # queries
        + 2 * _canvas_side_projection_flops(config, num_canvas_tokens=num_canvas)  # keys, values
        + attention_flops(num_queries=num_glimpse, num_keys=num_canvas, dim=canvas_dim)
        + linear_flops(num_tokens=num_glimpse, in_dim=canvas_dim, out_dim=backbone_dim)  # output
    )


def canvas_attention_write_flops(config: CanViTConfig, *, glimpse_size_px: int, canvas_grid_size: int) -> int:
    """One Canvas Attention Write: canvas tokens query the glimpse."""
    num_glimpse = num_glimpse_tokens(config, glimpse_size_px=glimpse_size_px)
    num_canvas = num_canvas_tokens(config, canvas_grid_size=canvas_grid_size)
    backbone_dim, canvas_dim = config.backbone_spec.embed_dim, config.canvas_dim
    return (
        2 * _canvas_side_projection_flops(config, num_canvas_tokens=num_canvas)  # queries, output
        + 2 * linear_flops(num_tokens=num_glimpse, in_dim=backbone_dim, out_dim=canvas_dim)  # keys, values
        + attention_flops(num_queries=num_canvas, num_keys=num_glimpse, dim=canvas_dim)
    )


def glimpse_flops(config: CanViTConfig, *, glimpse_size_px: int, canvas_grid_size: int) -> int:
    """One CanViT forward: patch embedding, VPE, backbone blocks, Canvas Attention Reads and Writes."""
    spec = config.backbone_spec
    reads, writes = canvas_attention_schedule(
        num_blocks=spec.num_blocks, rw_stride=config.rw_stride, enable_reads=config.enable_reads,
    )
    # Random Fourier features of (row/scale, col/scale, log scale): a [3] × [3, dim/2] product.
    vpe = linear_flops(num_tokens=1, in_dim=3, out_dim=spec.embed_dim // 2) if config.enable_vpe else 0
    block = _vit_block_flops(
        num_tokens=num_glimpse_tokens(config, glimpse_size_px=glimpse_size_px),
        dim=spec.embed_dim, mlp_hidden_dim=spec.mlp_hidden_dim,
    )
    read = canvas_attention_read_flops(config, glimpse_size_px=glimpse_size_px, canvas_grid_size=canvas_grid_size)
    write = canvas_attention_write_flops(config, glimpse_size_px=glimpse_size_px, canvas_grid_size=canvas_grid_size)
    num_patches = _glimpse_grid_size(config, glimpse_size_px) ** 2
    return (
        _patch_embedding_flops(num_patches=num_patches, patch_size=spec.patch_size, dim=spec.embed_dim)
        + vpe
        + spec.num_blocks * block
        + len(reads) * read
        + len(writes) * write
    )


def segmentation_probe_flops(*, grid_size: int, embed_dim: int, num_classes: int) -> int:
    """A SegmentationProbe on a grid_size × grid_size feature map: its 1×1 convolution."""
    return linear_flops(num_tokens=grid_size**2, in_dim=embed_dim, out_dim=num_classes)


def dinov3_flops(config: "DINOv3ViTConfig", *, input_size_px: int) -> int:
    """A DINOv3 ViT on a square RGB image, from its Hugging Face config."""
    patch_size = config.patch_size
    assert isinstance(patch_size, int) and config.num_channels == 3 and not config.use_gated_mlp, (
        "modeled: DINOv3 ViTs with square patches of RGB pixels and a two-layer MLP"
    )
    assert input_size_px % patch_size == 0, f"{input_size_px} px is not a whole number of {patch_size} px patches"
    num_patches = (input_size_px // patch_size) ** 2
    block = _vit_block_flops(
        num_tokens=1 + config.num_register_tokens + num_patches,  # CLS, registers, patches
        dim=config.hidden_size, mlp_hidden_dim=config.intermediate_size,
    )
    return (
        _patch_embedding_flops(num_patches=num_patches, patch_size=patch_size, dim=config.hidden_size)
        + config.num_hidden_layers * block
    )


def _canvas_side_projection_flops(config: CanViTConfig, *, num_canvas_tokens: int) -> int:
    """One learned projection of the canvas tokens; the asymmetric design has none."""
    match config.canvas_projections:
        case "asymmetric":
            return 0
        case "qkvo":
            return linear_flops(num_tokens=num_canvas_tokens, in_dim=config.canvas_dim, out_dim=config.canvas_dim)
        case unreachable:
            assert_never(unreachable)


def _vit_block_flops(*, num_tokens: int, dim: int, mlp_hidden_dim: int) -> int:
    return (
        linear_flops(num_tokens=num_tokens, in_dim=dim, out_dim=3 * dim)  # queries, keys, values
        + attention_flops(num_queries=num_tokens, num_keys=num_tokens, dim=dim)
        + linear_flops(num_tokens=num_tokens, in_dim=dim, out_dim=dim)  # output
        + 2 * linear_flops(num_tokens=num_tokens, in_dim=dim, out_dim=mlp_hidden_dim)  # the MLP's two layers
    )


def _patch_embedding_flops(*, num_patches: int, patch_size: int, dim: int) -> int:
    """A convolution with stride equal to its kernel: one linear map from each RGB patch."""
    return linear_flops(num_tokens=num_patches, in_dim=3 * patch_size**2, out_dim=dim)


def _glimpse_grid_size(config: CanViTConfig, glimpse_size_px: int) -> int:
    patch_size = config.backbone_spec.patch_size
    assert glimpse_size_px % patch_size == 0, f"{glimpse_size_px} px is not a whole number of {patch_size} px patches"
    return glimpse_size_px // patch_size
