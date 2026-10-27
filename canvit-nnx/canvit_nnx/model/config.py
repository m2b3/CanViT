from typing import Literal

from canvit_core.config import CanvasProjections, CanViTConfig

ComputeDtype = Literal["float32", "bfloat16"]
"""The dtype of CanViT's activations, matrix products and recurrent state.

Parameters stay float32 and are cast where they are used. LayerNorm, RoPE and
the viewpoint encoding compute in float32, and attention accumulates its
scores, softmax and weighted sums in float32. "bfloat16" is the mixed precision
of CanViT-B's JAX fine-tuning on TPU.
"""

__all__ = ["CanViTConfig", "CanvasProjections", "ComputeDtype"]
