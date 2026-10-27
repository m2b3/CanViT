import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from canvit_pytorch import CanViT, CanViTForSemanticSegmentation, SegmentationProbe
from canvit_pytorch.hub.repos import Q_LEARNING_POLICIES, REINFORCE_POLICIES
from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.rl import LearnedPolicy, Observer, ViewpointScorer
from canvit_pytorch.rl.candidates import CandidateViewpoints
from canvit_pytorch.rl.features import FEATURE_GROUPS, CanvasReading, EpisodeFeatures, FeatureGroup, feature_channels
from canvit_pytorch.viewpoint import Viewpoint
from tests.model.tiny import GLIMPSE_PX, TINY

CONFIG_KEYS = {"blocks_per_stage", "canvas_dim", "candidates", "dueling", "feature_groups", "observer", "width"}

# The candidates the published policies chose at glimpses 1-4 on test_data/Cat03.jpg, recorded with the code that
# trained them (float32, CPU). Smallest top-two score margin: 0.0016 (Q-learning), 1.3 (REINFORCE).
CAT_CHOICES = {Q_LEARNING_POLICIES[0]: [205, 270, 0, 208], REINFORCE_POLICIES[0]: [14, 0, 255, 240]}

TINY_OBSERVER = Observer(
    pretrained_repo="unused", probe_repo="unused", scene_size_px=64, glimpse_size_px=GLIMPSE_PX, canvas_grid_size=4,
)


def small_scorer(
    *, canvas_dim: int, dueling: bool, feature_groups: tuple[FeatureGroup, ...] = FEATURE_GROUPS,
) -> ViewpointScorer:
    return ViewpointScorer(
        observer=TINY_OBSERVER, canvas_dim=canvas_dim, feature_groups=feature_groups,
        candidates=CandidateViewpoints(scales=(0.5, 0.25), centers_per_axis=4), width=8, blocks_per_stage=1,
        dueling=dueling,
    ).eval()


def episode_viewpoints(policy: LearnedPolicy, images: torch.Tensor, *, num_glimpses: int) -> Viewpoint:
    with torch.inference_mode():
        steps = policy.rollout(images, num_glimpses=num_glimpses)
    return Viewpoint(centers=torch.cat([step.viewpoint.centers for step in steps]),
                     scales=torch.cat([step.viewpoint.scales for step in steps]))


@pytest.mark.parametrize(
    ("dueling", "feature_groups"), [(True, FEATURE_GROUPS), (False, FEATURE_GROUPS), (True, ("patches",))],
)
def test_saved_scorer_reloads_identically(
    tmp_path: Path, dueling: bool, feature_groups: tuple[FeatureGroup, ...],
) -> None:
    scorer = small_scorer(canvas_dim=8, dueling=dueling, feature_groups=feature_groups)
    scorer.save_pretrained(tmp_path)
    assert set(json.loads((tmp_path / "config.json").read_text())) == CONFIG_KEYS
    loaded = ViewpointScorer.from_pretrained(str(tmp_path)).eval()
    assert (loaded.observer, loaded.candidates) == (scorer.observer, scorer.candidates)
    assert loaded.feature_groups == feature_groups
    features = torch.randn(2, sum(feature_channels(feature_groups, 8)), 32, 32)
    scores = scorer(features)
    assert scores.shape == (2, 2 * 4 * 4)  # [B, num_scales × centers_per_axis²]
    assert torch.equal(scores, loaded(features))


def test_features_compare_with_the_previous_and_the_initial_canvas() -> None:
    initial, previous, current = (
        CanvasReading(logits=torch.randn(1, 5, 32, 32), entropy=torch.rand(1, 32, 32), patches=torch.randn(1, 8, 32, 32))
        for _ in range(3)
    )
    episode = EpisodeFeatures(initial, FEATURE_GROUPS)
    episode(previous)
    full = episode(current)
    _, entropy_change, to_previous, to_initial, _, patch_change = full.split(feature_channels(FEATURE_GROUPS, 8), dim=1)

    def dissimilarity(a: CanvasReading, b: CanvasReading) -> torch.Tensor:
        return 1 - torch.cosine_similarity(a.patches, b.patches, dim=1).unsqueeze(1)

    torch.testing.assert_close(entropy_change, (current.entropy - previous.entropy).unsqueeze(1))
    torch.testing.assert_close(to_previous, dissimilarity(current, previous))
    torch.testing.assert_close(to_initial, dissimilarity(current, initial))
    torch.testing.assert_close(patch_change, current.patches - previous.patches)

    selection: tuple[FeatureGroup, ...] = ("patch_change", "entropy")
    selected = EpisodeFeatures(initial, selection)
    selected(previous)
    channels = dict(zip(FEATURE_GROUPS, full.split(feature_channels(FEATURE_GROUPS, 8), dim=1), strict=True))
    torch.testing.assert_close(selected(current), torch.cat([channels[group] for group in selection], dim=1))


def test_policy_starts_afresh_with_each_episode() -> None:
    torch.manual_seed(0)
    canvit = CanViT(TINY).eval()
    probe = SegmentationProbe(embed_dim=canvit.canvas_dim, num_classes=5, dropout=0.0, use_ln=True).eval()
    model = CanViTForSemanticSegmentation(canvit=canvit, probe=probe)
    policy = LearnedPolicy(scorer=small_scorer(canvas_dim=canvit.canvas_dim, dueling=True), model=model)
    scenes = torch.randn(3, 3, 64, 64)
    first = episode_viewpoints(policy, scenes[:1], num_glimpses=4)
    episode_viewpoints(policy, scenes[1:], num_glimpses=4)
    again = episode_viewpoints(policy, scenes[:1], num_glimpses=4)
    assert torch.equal(first.centers, again.centers) and torch.equal(first.scales, again.scales)
    assert (first.centers.abs() + first.scales.unsqueeze(1) <= 1 + 1e-6).all(), "every glimpse lies inside the scene"


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("repo", list(CAT_CHOICES))
def test_published_policies_choose_as_when_trained(repo: str) -> None:
    policy = LearnedPolicy.from_pretrained(repo, torch.device("cpu"))
    image = preprocess(policy.scorer.observer.scene_size_px)(Image.open("test_data/Cat03.jpg").convert("RGB"))
    assert isinstance(image, torch.Tensor)
    chosen = episode_viewpoints(policy, image.unsqueeze(0), num_glimpses=5)
    expected = policy.scorer.candidates.viewpoints(torch.tensor(CAT_CHOICES[repo]))
    assert torch.equal(chosen.centers[1:], expected.centers) and torch.equal(chosen.scales[1:], expected.scales)
