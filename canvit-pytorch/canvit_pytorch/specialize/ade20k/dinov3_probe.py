"""Probes on the patch features of a frozen DINOv3 ViT, the passive baseline: one view of the whole scene."""

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor

from canvit_pytorch.benchmarks.ade20k.dataset import Split
from canvit_pytorch.specialize.ade20k.config import ProbeTrainingConfig
from canvit_pytorch.teacher import DINOV3_REPOS, DINOv3Teacher, DINOv3Variant, load_teacher


@dataclass(frozen=True)
class DINOv3ProbeConfig:
    """Train an ADE20K probe on the patch features of a frozen DINOv3 ViT."""

    training: ProbeTrainingConfig
    variant: DINOv3Variant = "vitb16"
    input_size_px: int = 128
    """The scene is resized to input_size_px × input_size_px before DINOv3 sees it."""

    @property
    def run_name(self) -> str:
        return f"dinov3-probe_{self.variant}_{self.input_size_px}px"

    def load_feature_maps(self, device: torch.device) -> "DINOv3FeatureMaps":
        return DINOv3FeatureMaps(dinov3=load_teacher(DINOV3_REPOS[self.variant], device), input_size_px=self.input_size_px)


@dataclass(frozen=True)
class DINOv3FeatureMaps:
    dinov3: DINOv3Teacher
    input_size_px: int
    layer_normalized: ClassVar[bool] = True  # DINOv3's outputs pass through its final LayerNorm

    def __post_init__(self) -> None:
        patch_size = self.dinov3.patch_size
        assert self.input_size_px % patch_size == 0, f"input_size_px={self.input_size_px} is not a multiple of {patch_size}"

    @property
    def embed_dim(self) -> int:
        return self.dinov3.embed_dim

    @torch.no_grad()
    def __call__(self, images: Tensor, split: Split) -> list[Tensor]:
        """One map, the patch features [B, g, g, D] of the bilinearly resized scene; the same for both splits."""
        size = self.input_size_px
        resized = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
        grid = size // self.dinov3.patch_size
        return [self.dinov3(resized).patches.view(images.shape[0], grid, grid, -1)]
