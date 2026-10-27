"""Training episodes: the scorer's decisions under the objective's exploration, and the reward of each glimpse."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from canvit_pytorch.episode import take_glimpse
from canvit_pytorch.model.canvit import RecurrentState
from canvit_pytorch.model.segmentation import CanViTForSemanticSegmentation
from canvit_pytorch.rl.features import CanvasReading, EpisodeFeatures, read_canvas
from canvit_pytorch.rl.scorer import ViewpointScorer
from canvit_pytorch.rl.train.reward import fraction_removed, segmentation_cross_entropy
from canvit_pytorch.viewpoint import Viewpoint


@dataclass(frozen=True)
class Decisions:
    """The decisions of a batch of episodes, decision-major: the scorer's input and the candidate taken at each
    decision, and the reward of the glimpse taken there."""

    features: Tensor  # [num_decisions * B, C, G, G]
    taken: Tensor  # [num_decisions * B]
    rewards: Tensor  # [num_decisions, B] the fraction of the probe's cross-entropy each glimpse removed


@torch.no_grad()
def explore(
    model: CanViTForSemanticSegmentation, scorer: ViewpointScorer, images: Tensor, labels: Tensor, *,
    choose: Callable[[Tensor], Tensor], num_glimpses: int, reward_resolution_px: int, amp: bool,
) -> Decisions:
    """An episode on each scene, with the scorer's observer: the full scene, then num_glimpses - 1 candidates that
    `choose` picks from the scores [B, num_candidates]. The reward's cross-entropy is computed at
    reward_resolution_px; CanViT runs under bfloat16 autocast with amp."""
    assert not scorer.training, "the scorer must be in eval mode, so its BatchNorm uses running statistics"
    observer = scorer.observer
    batch_size = images.shape[0]

    def observe(state: RecurrentState, viewpoint: Viewpoint) -> tuple[RecurrentState, CanvasReading, Tensor]:
        with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=amp):
            state = take_glimpse(
                canvit=model.canvit, images=images, state=state, viewpoint=viewpoint,
                glimpse_size_px=observer.glimpse_size_px,
            ).state
        reading = read_canvas(model, state.canvas)
        return state, reading, segmentation_cross_entropy(reading.logits, labels, resolution_px=reward_resolution_px)

    state = model.init_state(batch_size=batch_size, canvas_grid_size=observer.canvas_grid_size)
    episode = EpisodeFeatures(read_canvas(model, state.canvas))
    state, reading, loss = observe(state, Viewpoint.full_scene(batch_size=batch_size, device=images.device))
    features, taken, losses = [], [], [loss]
    for _ in range(num_glimpses - 1):
        features.append(episode(reading))
        taken.append(choose(scorer(features[-1])))
        state, reading, loss = observe(state, scorer.candidates.viewpoints(taken[-1]))
        losses.append(loss)
    stacked = torch.stack(losses)
    return Decisions(
        features=torch.cat(features), taken=torch.cat(taken), rewards=fraction_removed(stacked[:-1], stacked[1:]),
    )
