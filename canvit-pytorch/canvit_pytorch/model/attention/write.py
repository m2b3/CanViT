from typing import final

from torch import nn

from canvit_pytorch.model.attention.base import CanvasAttention


@final
class CanvasAttentionWrite(CanvasAttention):
    """Canvas tokens query the glimpse. Learned projections sit on the glimpse side only."""

    def __init__(self, *, backbone_dim: int, canvas_dim: int, num_heads: int) -> None:
        super().__init__(query_dim=canvas_dim, kv_dim=backbone_dim, canvas_dim=canvas_dim, num_heads=num_heads)
        self.k_map = nn.Linear(backbone_dim, canvas_dim)
        self.v_map = nn.Linear(backbone_dim, canvas_dim)
