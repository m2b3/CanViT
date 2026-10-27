"""The frozen CanViT and segmentation probe a learned policy reads, and how they view scenes."""

from dataclasses import dataclass

import torch

from canvit_pytorch.hub import repos
from canvit_pytorch.model.segmentation import CanViTForSemanticSegmentation


@dataclass(frozen=True, kw_only=True)
class Observer:
    """A pretrained CanViT and a segmentation probe trained on its canvas, viewing scenes of scene_size_px through
    glimpses of glimpse_size_px on a canvas_grid_size × canvas_grid_size canvas. The checkpoints are Hub ids, which
    load from a local mirror when $CANVIT_HUB_ROOT names one (hub.repos.resolve), or local directories.

    A ViewpointScorer records the observer it was trained with, and its policy runs with that observer.
    """

    pretrained_repo: str
    probe_repo: str
    scene_size_px: int
    glimpse_size_px: int
    canvas_grid_size: int

    def load(self, device: torch.device) -> CanViTForSemanticSegmentation:
        """The CanViT and probe, frozen, on device."""
        model = CanViTForSemanticSegmentation.from_pretrained_with_probe(
            pretrained_repo=repos.resolve(self.pretrained_repo), probe_repo=repos.resolve(self.probe_repo),
        )
        return model.to(device).eval().requires_grad_(False)


def flagship_observer(*, canvas_grid_size: int) -> Observer:
    """The flagship CanViT-B and its released ADE20K probe for that canvas grid, by their Hub ids, at the released
    scene and glimpse sizes."""
    pretrained, probe = repos.released_segmenter(
        scene_size_px=repos.RELEASED_SCENE_SIZE_PX, canvas_grid_size=canvas_grid_size,
    )
    return Observer(
        pretrained_repo=repos.hub_id(pretrained), probe_repo=repos.hub_id(probe),
        scene_size_px=repos.RELEASED_SCENE_SIZE_PX, glimpse_size_px=repos.RELEASED_GLIMPSE_SIZE_PX,
        canvas_grid_size=canvas_grid_size,
    )
