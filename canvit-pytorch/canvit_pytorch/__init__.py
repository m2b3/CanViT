"""CanViT, the Canvas Vision Transformer: an active-vision foundation model that sees a scene
through a sequence of glimpses and remembers it on a scene-wide canvas."""

from canvit_pytorch.model import (
    CanViT,
    CanViTConfig,
    CanViTForImageClassification,
    CanViTForPretraining,
    CanViTForSemanticSegmentation,
    CanViTOutput,
    RecurrentState,
)
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint

__all__ = [
    "CanViT",
    "CanViTConfig",
    "CanViTForImageClassification",
    "CanViTForPretraining",
    "CanViTForSemanticSegmentation",
    "CanViTOutput",
    "RecurrentState",
    "SegmentationProbe",
    "Viewpoint",
    "sample_at_viewpoint",
]


def __getattr__(name: str) -> object:
    from canvit_pytorch import legacy

    raise legacy.attribute_error(__name__, name)
