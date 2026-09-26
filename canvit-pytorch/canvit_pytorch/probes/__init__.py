"""Lightweight probe heads for downstream tasks.

Probe architectures are first-class citizens of canvit-pytorch (rather than
satellite repos) so they can be (a) composed into HF model wrappers like
:class:`CanViTForSemanticSegmentation` and (b) consumed by canvit-eval
without pulling in canvit-specialize' training-only dependencies.

The trainer that fits these probes still lives in canvit-specialize; only the
architecture + HF Hub I/O lives here.
"""

from canvit_pytorch.probes.segmentation import SegmentationProbe

__all__ = ["SegmentationProbe"]
