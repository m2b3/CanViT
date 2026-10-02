from dataclasses import dataclass
from typing import Literal, get_args

from canvit_core.backbone import BACKBONES, BackboneName, ViTSpec

CanvasProjections = Literal["asymmetric", "qkvo"]


@dataclass(frozen=True)
class CanViTConfig:
    backbone_name: BackboneName = "vitb16"
    canvas_num_heads: int = 8
    canvas_head_dim: int = 128
    num_canvas_registers: int = 16
    num_backbone_registers: int = 5
    rw_stride: int = 2
    enable_reads: bool = True
    enable_vpe: bool = True
    canvas_projections: CanvasProjections = "asymmetric"

    def __post_init__(self) -> None:
        dimensions = (
            self.canvas_num_heads,
            self.canvas_head_dim,
            self.rw_stride,
            self.num_canvas_registers,
            self.num_backbone_registers,
        )
        assert all(type(value) is int for value in dimensions), self
        assert isinstance(self.enable_reads, bool) and isinstance(self.enable_vpe, bool), self
        assert self.backbone_name in BACKBONES, f"backbone {self.backbone_name!r} not in {sorted(BACKBONES)}"
        assert self.canvas_projections in get_args(CanvasProjections), self.canvas_projections
        assert min(self.canvas_num_heads, self.canvas_head_dim, self.rw_stride) >= 1, self
        assert self.canvas_head_dim % 4 == 0, "2D RoPE needs canvas_head_dim divisible by 4"
        assert min(self.num_canvas_registers, self.num_backbone_registers) >= 0, self

    @property
    def backbone_spec(self) -> ViTSpec:
        return BACKBONES[self.backbone_name]

    @property
    def canvas_dim(self) -> int:
        return self.canvas_num_heads * self.canvas_head_dim
