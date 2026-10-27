"""ViewpointScorer: a score for every candidate viewpoint, from the features of the canvas (rl.features).

FeatureFusion merges the feature groups into a map, which a U-Net pools down to a single cell and expands back, so
that every score depends on the whole scene. Each candidate's score is read from the map at its center, plus an
embedding of its scale, by a head shared across scales. A dueling scorer's scores are action values: a value of
the scene, from its mean features, plus advantages of mean zero over the candidates.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from canvit_pytorch.hub.loading import HubMixin
from canvit_pytorch.rl.candidates import CandidateViewpoints
from canvit_pytorch.rl.features import FEATURE_GRID_SIZE, feature_channels
from canvit_pytorch.rl.observer import Observer
from canvit_pytorch.rl.scorer.convnext import ConvNeXtV2Stage
from canvit_pytorch.rl.scorer.fusion import FeatureFusion
from canvit_pytorch.rl.scorer.unet import UNet


class ViewpointScorer(nn.Module, HubMixin):
    """Features [B, C, FEATURE_GRID_SIZE, FEATURE_GRID_SIZE] (rl.features.EpisodeFeatures) -> scores
    [B, num_candidates], numbered as CandidateViewpoints numbers them.

    The keyword arguments are the checkpoint's config.json. canvas_dim is that of the observer's CanViT.
    """

    candidate_centers_xy: Tensor  # [num_scales, centers_per_axis, centers_per_axis, 2] as grid_sample's (x, y)

    def __init__(
        self, *, observer: Observer, canvas_dim: int, candidates: CandidateViewpoints, width: int,
        blocks_per_stage: int, dueling: bool,
    ) -> None:
        super().__init__()
        self.observer = observer
        self.candidates = candidates
        channels = feature_channels(canvas_dim)
        self.feature_fusion = FeatureFusion(channels, width)
        self.unet = UNet(grid_size=FEATURE_GRID_SIZE, width=width, blocks_per_stage=blocks_per_stage)
        self.register_buffer("candidate_centers_xy", candidates.grid()[..., :2].flip(-1), persistent=False)
        self.scale_embeddings = nn.Parameter(torch.zeros(len(candidates.scales), width, 1, 1))
        self.score_head = nn.Sequential(
            ConvNeXtV2Stage(width, width, blocks_per_stage), nn.Conv2d(width, 1, kernel_size=1),
        )
        self.value_head = nn.Sequential(nn.Linear(sum(channels), 64), nn.GELU(), nn.Linear(64, 1)) if dueling else None

    def forward(self, features: Tensor) -> Tensor:
        assert features.shape[-2:] == (FEATURE_GRID_SIZE, FEATURE_GRID_SIZE), features.shape
        x = self.unet(self.feature_fusion(features))
        batch_size = x.shape[0]
        scores = torch.cat([
            self.score_head(F.grid_sample(x, centers.expand(batch_size, -1, -1, -1), align_corners=False) + embedding)
            for centers, embedding in zip(self.candidate_centers_xy, self.scale_embeddings, strict=True)
        ], dim=1).flatten(1)
        if self.value_head is None:
            return scores
        # Advantages of mean zero, so that the value and the advantages are identifiable.
        return scores - scores.mean(dim=1, keepdim=True) + self.value_head(features.mean(dim=(2, 3)))
