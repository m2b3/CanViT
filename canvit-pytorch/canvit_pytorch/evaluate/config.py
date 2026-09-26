"""The paper's evaluation protocol: scene and glimpse sizes, and how CanViT views each scene."""

from dataclasses import dataclass

from torch import Tensor

from canvit_pytorch.episode import EpisodeStep, run_episode
from canvit_pytorch.hub.repos import RELEASED_CANVAS_GRID_SIZE, RELEASED_GLIMPSE_SIZE_PX, RELEASED_SCENE_SIZE_PX
from canvit_pytorch.model import CanViT
from canvit_pytorch.policies import POLICIES, PolicyName, make_policy
from canvit_pytorch.policies.entropy import CanvasLogits

SCENE_SIZE_PX = RELEASED_SCENE_SIZE_PX
"""Side of the square scene that ADE20K and ImageNet-1k images are resized to, as in pretraining."""
GLIMPSE_SIZE_PX = RELEASED_GLIMPSE_SIZE_PX
CANVAS_GRID_SIZE = RELEASED_CANVAS_GRID_SIZE
"""The canvas grid CanViT-B was pretrained with: one canvas patch per 16 px of a 512 px scene."""
NUM_GLIMPSES = 21
"""Glimpses per scene: the full scene, its 4 quadrants and its 16 sixteenths under C2F."""


@dataclass(frozen=True)
class EpisodeConfig:
    """How CanViT views each scene: the viewing policy, the number of glimpses, the glimpse and canvas sizes."""

    policy: PolicyName = "coarse_to_fine"
    num_glimpses: int = NUM_GLIMPSES
    canvas_grid_size: int = CANVAS_GRID_SIZE
    glimpse_size_px: int = GLIMPSE_SIZE_PX

    def __post_init__(self) -> None:
        assert self.num_glimpses >= 1, self
        spec = POLICIES[self.policy]
        assert spec.supports_canvas_grid(self.canvas_grid_size), (
            f"{spec.paper_name} does not support a {self.canvas_grid_size}×{self.canvas_grid_size} canvas"
        )

    def rollout(
        self, *, canvit: CanViT, images: Tensor, canvas_logits: CanvasLogits | None = None,
    ) -> list[EpisodeStep]:
        """One episode per ImageNet-normalized scene in images [B, 3, H, W], starting from an empty canvas.

        canvas_logits decodes a canvas into segmentation logits; EG-C2F needs it, other policies ignore it.
        """
        batch_size = images.shape[0]
        policy = make_policy(
            self.policy, batch_size=batch_size, device=images.device, num_glimpses=self.num_glimpses,
            canvas_grid_size=self.canvas_grid_size, canvas_logits=canvas_logits,
        )
        return run_episode(
            canvit=canvit, images=images, policy=policy, num_glimpses=self.num_glimpses,
            glimpse_size_px=self.glimpse_size_px,
            initial_state=canvit.init_state(batch_size=batch_size, canvas_grid_size=self.canvas_grid_size),
        )
