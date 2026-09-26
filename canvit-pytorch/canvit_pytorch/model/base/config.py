"""CanViT configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class CanViTConfig:
    """CanViT configuration."""

    rw_stride: int = 2
    enable_reads: bool = True
    n_backbone_registers: int = 5
    n_canvas_registers: int = 16
    canvas_num_heads: int = 8
    canvas_head_dim: int = 128
    enable_vpe: bool = True
    canvas_update_mode: Literal["additive", "convex"] = "additive"
    canvas_proj_mode: Literal["asymmetric", "full"] = "asymmetric"
    gate_bias_init: float | None = None

    def __post_init__(self) -> None:
        is_convex = self.canvas_update_mode == "convex"
        has_gate = self.gate_bias_init is not None
        assert is_convex == has_gate, (
            f"Inconsistent config: canvas_update_mode={self.canvas_update_mode!r}, "
            f"gate_bias_init={self.gate_bias_init!r}"
        )

    @property
    def canvas_dim(self) -> int:
        return self.canvas_num_heads * self.canvas_head_dim
