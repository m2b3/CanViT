"""Run a CanViT episode: T glimpses sampled by a policy, recurrent state updated each step."""

from dataclasses import dataclass
from typing import Protocol

from canvit_pytorch import CanViTOutput, RecurrentState, Viewpoint, sample_at_viewpoint
from torch import Tensor


class CanViTModel(Protocol):
    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState: ...
    def __call__(self, *, glimpse: Tensor, state: RecurrentState, viewpoint: Viewpoint) -> CanViTOutput: ...


class Policy(Protocol):
    def step(self, t: int, state: RecurrentState) -> Viewpoint: ...


@dataclass(frozen=True)
class EpisodeStep:
    t: int
    state: RecurrentState
    output: CanViTOutput
    viewpoint: Viewpoint


def run_episode(
    *,
    model: CanViTModel,
    images: Tensor,
    policy: Policy,
    n_timesteps: int,
    canvas_grid: int,
    glimpse_px: int,
    state: RecurrentState | None = None,
) -> list[EpisodeStep]:
    B = images.shape[0]
    if state is None:
        state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)

    steps: list[EpisodeStep] = []
    for t in range(n_timesteps):
        vp = policy.step(t, state)
        glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=glimpse_px)
        out = model(glimpse=glimpse, state=state, viewpoint=vp)
        state = out.state
        steps.append(EpisodeStep(t=t, state=state, output=out, viewpoint=vp))

    return steps
