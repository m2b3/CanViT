from dataclasses import dataclass
from typing import get_args

from canvit_pytorch.model.attention import CanvasProjections
from canvit_pytorch.model.backbone import BACKBONES, BackboneName, ViTSpec


@dataclass(frozen=True)
class CanViTConfig:
    """CanViT's architecture. The defaults are CanViT-B; each paper ablation changes one field."""

    backbone_name: BackboneName = "vitb16"
    canvas_num_heads: int = 8
    canvas_head_dim: int = 128
    num_canvas_registers: int = 16
    num_backbone_registers: int = 5
    rw_stride: int = 2
    """Backbone blocks between consecutive Canvas Attention operations, which alternate Read and Write."""
    enable_reads: bool = True
    """Without Reads the canvas is write-only: the backbone never sees it."""
    enable_vpe: bool = True
    """Add a Viewpoint Encoding token to the glimpse stream."""
    canvas_projections: CanvasProjections = "asymmetric"
    """"qkvo" also learns canvas-side projections in Canvas Attention."""

    def __post_init__(self) -> None:
        assert self.backbone_name in BACKBONES, f"backbone {self.backbone_name!r} not in {sorted(BACKBONES)}"
        assert self.canvas_projections in get_args(CanvasProjections), self.canvas_projections
        assert min(self.canvas_num_heads, self.canvas_head_dim, self.rw_stride) >= 1, self
        assert min(self.num_canvas_registers, self.num_backbone_registers) >= 0, self

    @property
    def backbone_spec(self) -> ViTSpec:
        return BACKBONES[self.backbone_name]

    @property
    def canvas_dim(self) -> int:
        return self.canvas_num_heads * self.canvas_head_dim
