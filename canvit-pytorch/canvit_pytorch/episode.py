"""Rollouts: a policy chooses each viewpoint, CanViT processes the glimpse taken there."""

from dataclasses import dataclass

from torch import Tensor

from canvit_pytorch.model.canvit import CanViT, CanViTOutput, RecurrentState
from canvit_pytorch.policies import Policy
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint


@dataclass(frozen=True)
class EpisodeStep:
    t: int
    viewpoint: Viewpoint
    output: CanViTOutput

    @property
    def state(self) -> RecurrentState:
        return self.output.state


def run_episode(
    *,
    canvit: CanViT,
    images: Tensor,
    policy: Policy,
    num_glimpses: int,
    glimpse_size_px: int,
    initial_state: RecurrentState,
) -> list[EpisodeStep]:
    """Take num_glimpses glimpses of ImageNet-normalized images [B, 3, H, W]."""
    steps: list[EpisodeStep] = []
    state = initial_state
    for t in range(num_glimpses):
        viewpoint = policy.step(t, state)
        glimpse = sample_at_viewpoint(spatial=images, viewpoint=viewpoint, glimpse_size_px=glimpse_size_px)
        output = canvit(glimpse=glimpse, state=state, viewpoint=viewpoint)
        steps.append(EpisodeStep(t=t, viewpoint=viewpoint, output=output))
        state = output.state
    return steps
