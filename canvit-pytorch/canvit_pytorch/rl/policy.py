"""LearnedPolicy: the viewing policy of a trained ViewpointScorer, a canvit_pytorch.policies.Policy."""

import torch
from torch import Tensor

from canvit_pytorch.episode import EpisodeStep, run_episode
from canvit_pytorch.model.canvit import RecurrentState
from canvit_pytorch.model.segmentation import CanViTForSemanticSegmentation
from canvit_pytorch.rl.features import EpisodeFeatures, read_canvas
from canvit_pytorch.rl.scorer import ViewpointScorer
from canvit_pytorch.viewpoint import Viewpoint


class LearnedPolicy:
    """The full scene first; then, at each glimpse, the candidate viewpoint of highest score.

    The scorer reads the canvas through `model`, the CanViT and probe of the observer it was trained with. Its
    features compare the canvas with the state before the first glimpse, so episodes start from the initial
    state (rollout), and a policy runs one episode at a time.
    """

    def __init__(self, *, scorer: ViewpointScorer, model: CanViTForSemanticSegmentation) -> None:
        self.scorer = scorer
        self.model = model
        self.episode_features: EpisodeFeatures | None = None

    @classmethod
    def from_pretrained(cls, scorer_repo: str, device: torch.device, *, revision: str | None = None) -> "LearnedPolicy":
        """The scorer of a Hub repo (at revision, the latest when None) or local directory, with the observer it
        records, on device."""
        scorer = ViewpointScorer.from_pretrained(scorer_repo, revision=revision).to(device).eval()
        return cls(scorer=scorer, model=scorer.observer.load(device))

    @torch.no_grad()
    def step(self, t: int, state: RecurrentState) -> Viewpoint:
        assert not self.scorer.training, "the scorer must be in eval mode, so its BatchNorm uses running statistics"
        canvas = state.canvas
        with torch.autocast(canvas.device.type, enabled=False):
            reading = read_canvas(self.model, canvas)
            if t == 0:
                self.episode_features = EpisodeFeatures(reading, self.scorer.feature_groups)
                return Viewpoint.full_scene(batch_size=canvas.shape[0], device=canvas.device)
            assert self.episode_features is not None, "an episode starts at t = 0"
            scores = self.scorer(self.episode_features(reading))
        return self.scorer.candidates.viewpoints(scores.argmax(dim=1))

    def rollout(self, images: Tensor, *, num_glimpses: int) -> list[EpisodeStep]:
        """An episode of num_glimpses glimpses on each ImageNet-normalized scene [B, 3, S, S], S the observer's
        scene size."""
        observer = self.scorer.observer
        assert images.shape[-2:] == (observer.scene_size_px, observer.scene_size_px), (images.shape, observer)
        return run_episode(
            canvit=self.model.canvit, images=images, policy=self, num_glimpses=num_glimpses,
            glimpse_size_px=observer.glimpse_size_px,
            initial_state=self.model.init_state(batch_size=images.shape[0], canvas_grid_size=observer.canvas_grid_size),
        )
