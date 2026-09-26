"""Lightweight probe heads for downstream tasks.

Probe architectures live in the model package so they can be (a) composed
into HF model wrappers like :class:`CanViTForSemanticSegmentation` and (b)
consumed by :mod:`canvit_pytorch.evaluate` without the training-only
dependencies of :mod:`canvit_pytorch.specialize`, whose trainer fits them.
"""

from canvit_pytorch.probes.segmentation import SegmentationProbe

__all__ = ["SegmentationProbe"]
