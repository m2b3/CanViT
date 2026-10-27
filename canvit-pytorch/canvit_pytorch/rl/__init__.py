"""Learned viewing policies for a frozen CanViT and its segmentation probe (docs/rl.md).

A ViewpointScorer scores a fixed set of candidate viewpoints from features of the canvas and of the probe's
segmentation of it; a LearnedPolicy takes the candidate of highest score as its next viewpoint. A scorer records
the Observer it was trained with. The published scorers are hub.repos.Q_LEARNING_POLICIES and
hub.repos.REINFORCE_POLICIES; rl.train trains new ones.
"""

from canvit_pytorch.rl.observer import Observer
from canvit_pytorch.rl.policy import LearnedPolicy
from canvit_pytorch.rl.scorer import ViewpointScorer

__all__ = ["LearnedPolicy", "Observer", "ViewpointScorer"]
