"""Canvas Attention: the only way glimpse tokens and canvas tokens interact.

Reads let glimpse tokens query the canvas; Writes let canvas tokens query the
glimpse. The canvas side gets no learned projection (only LayerNorm and RoPE),
so the cost of a Read/Write pair grows linearly with the number of canvas tokens.
"""

from canvit_core.config import CanvasProjections

from canvit_pytorch.model.attention.base import CanvasAttention
from canvit_pytorch.model.attention.qkvo import CanvasAttentionReadQKVO, CanvasAttentionWriteQKVO
from canvit_pytorch.model.attention.read import CanvasAttentionRead
from canvit_pytorch.model.attention.write import CanvasAttentionWrite

CANVAS_ATTENTION_CLASSES: dict[CanvasProjections, tuple[type[CanvasAttention], type[CanvasAttention]]] = {
    "asymmetric": (CanvasAttentionRead, CanvasAttentionWrite),
    "qkvo": (CanvasAttentionReadQKVO, CanvasAttentionWriteQKVO),
}

__all__ = ["CANVAS_ATTENTION_CLASSES", "CanvasAttention", "CanvasProjections", "CanvasAttentionRead", "CanvasAttentionWrite"]
