"""Probes on the canvas of a frozen pretrained CanViT: one probe decodes the canvas after every glimpse."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, assert_never

import torch
from torch import Tensor

from canvit_pytorch import CanViT, CanViTForPretraining
from canvit_pytorch.benchmarks.ade20k.dataset import Split
from canvit_pytorch.episode import run_episode
from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_CANVAS_GRID_SIZE, RELEASED_GLIMPSE_SIZE_PX
from canvit_pytorch.policies import PolicyName, make_policy
from canvit_pytorch.specialize.ade20k.config import ProbeTrainingConfig


@dataclass(frozen=True)
class CanvasProbeConfig:
    """Train an ADE20K probe on the canvas of a frozen pretrained CanViT."""

    training: ProbeTrainingConfig
    pretrained_repo: str = FLAGSHIP
    """A CanViTForPretraining checkpoint: Hub repo id or local directory."""
    canvas_grid_size: int = RELEASED_CANVAS_GRID_SIZE
    glimpse_size_px: int = RELEASED_GLIMPSE_SIZE_PX
    num_glimpses: int = 10
    training_policy: PolicyName = "random"
    """The viewing policy of training episodes: R-IID in the paper."""
    validation_policy: PolicyName = "full_then_random"
    """The viewing policy of validation episodes: F-IID in the paper."""

    @property
    def run_name(self) -> str:
        return (f"canvas-probe_{Path(self.pretrained_repo).name}_s{self.training.scene_size_px}"
                f"_c{self.canvas_grid_size}_{self.num_glimpses}glimpses")

    def load_feature_maps(self, device: torch.device) -> "CanvasFeatureMaps":
        canvit = CanViTForPretraining.from_pretrained(self.pretrained_repo).canvit
        return CanvasFeatureMaps(canvit=canvit.to(device).eval().requires_grad_(False), config=self)


@dataclass(frozen=True)
class CanvasFeatureMaps:
    canvit: CanViT
    config: CanvasProbeConfig
    layer_normalized: ClassVar[bool] = False

    @property
    def embed_dim(self) -> int:
        return self.canvit.canvas_dim

    @torch.no_grad()
    def __call__(self, images: Tensor, split: Split) -> list[Tensor]:
        """The canvas patches [B, G, G, canvas_dim] after each glimpse of an episode on ImageNet-normalized images."""
        match split:
            case "training":
                policy_name = self.config.training_policy
            case "validation":
                policy_name = self.config.validation_policy
            case _:
                assert_never(split)
        batch_size, grid, num_glimpses = images.shape[0], self.config.canvas_grid_size, self.config.num_glimpses
        policy = make_policy(
            policy_name, batch_size=batch_size, device=images.device, num_glimpses=num_glimpses, canvas_grid_size=grid,
        )
        episode = run_episode(
            canvit=self.canvit, images=images, policy=policy, num_glimpses=num_glimpses,
            glimpse_size_px=self.config.glimpse_size_px,
            initial_state=self.canvit.init_state(batch_size=batch_size, canvas_grid_size=grid),
        )
        return [self.canvit.canvas_patch_grid(step.state.canvas) for step in episode]
