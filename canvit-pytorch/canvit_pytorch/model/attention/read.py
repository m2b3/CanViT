from typing import final

from torch import nn

from canvit_pytorch.model.attention.base import CanvasAttention


@final
class CanvasAttentionRead(CanvasAttention):
    """Glimpse tokens query the canvas. Learned projections sit on the glimpse side only."""

    def __init__(self, *, backbone_dim: int, canvas_dim: int, num_heads: int) -> None:
        super().__init__(query_dim=backbone_dim, kv_dim=canvas_dim, canvas_dim=canvas_dim, num_heads=num_heads)
        self.q_map = nn.Linear(backbone_dim, canvas_dim)
        self.o_map = nn.Linear(canvas_dim, backbone_dim)
