import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset

from canvit_pytorch import CanViTForPretraining, SegmentationProbe
from canvit_pytorch.benchmarks.ade20k.dataset import Split
from canvit_pytorch.rl import Observer, ViewpointScorer
from canvit_pytorch.rl.candidates import CandidateViewpoints
from canvit_pytorch.rl.train import PolicyTraining
from canvit_pytorch.rl.train.objectives import Reinforce
from canvit_pytorch.rl.train.reward import RunningStandardizer
from tests.model.tiny import GLIMPSE_PX, TINY


@dataclass(frozen=True, kw_only=True)
class TinyTraining(PolicyTraining):
    """The training loop on random 64 px scenes with labels of 5 classes."""

    def scenes(self, split: Split) -> Dataset[tuple[Tensor, Tensor]]:
        generator = torch.Generator().manual_seed(0 if split == "training" else 1)
        images, labels = torch.randn(4, 3, 64, 64, generator=generator), torch.randint(5, (4, 64, 64), generator=generator)
        return cast(Dataset[tuple[Tensor, Tensor]], TensorDataset(images, labels))


def tiny_observer(directory: Path) -> Observer:
    """A tiny CanViT and a 5-class probe, saved in directory."""
    torch.manual_seed(0)
    CanViTForPretraining(canvit_config=TINY, teacher_dim=8, teacher_patch_grid=4).save_pretrained(directory / "pretrained")
    SegmentationProbe(embed_dim=TINY.canvas_dim, num_classes=5, dropout=0.0, use_ln=True).save_pretrained(directory / "probe")
    return Observer(
        pretrained_repo=str(directory / "pretrained"), probe_repo=str(directory / "probe"), scene_size_px=64,
        glimpse_size_px=GLIMPSE_PX, canvas_grid_size=4,
    )


def test_training_selects_and_saves_a_scorer(tmp_path: Path) -> None:
    training = TinyTraining(
        out_dir=tmp_path / "run", observer=tiny_observer(tmp_path), feature_groups=("entropy", "patches"),
        candidates=CandidateViewpoints((0.5, 0.25), 4),
        width=8, blocks_per_stage=1, batch_size=2, glimpse_budget=30, reward_resolution_px=32, validation_interval=2,
        device="cpu", amp=False, num_workers=0,
    )
    scorer_dir = training.run()
    record = json.loads((tmp_path / "run" / "record.json").read_text())
    assert [validation["step"] for validation in record["validations"]] == [2, 3]
    cross_entropy = {v["step"]: v["validation_cross_entropy"] for v in record["validations"]}
    assert all(len(values) == training.num_glimpses for values in cross_entropy.values())
    after_first_glimpse = {step: sum(values[1:]) for step, values in cross_entropy.items()}
    assert record["selected_step"] == min(after_first_glimpse, key=after_first_glimpse.__getitem__)
    for directory in (scorer_dir, tmp_path / "run" / "last_step_scorer"):
        scorer = ViewpointScorer.from_pretrained(str(directory))
        assert (scorer.observer, scorer.feature_groups) == (training.observer, training.feature_groups)


def test_running_standardizer_is_unit_scale_from_the_first_batch() -> None:
    generator = torch.Generator().manual_seed(0)
    standardize = RunningStandardizer(num_streams=2, momentum=0.997, device=torch.device("cpu"))
    streams = torch.stack([5 + 2 * torch.randn(256, generator=generator), -3 + 0.1 * torch.randn(256, generator=generator)])
    first = standardize(streams)
    assert (first.mean(dim=1).abs() < 0.1).all() and ((first.std(dim=1) - 1).abs() < 0.1).all(), first


def test_entropy_bonus_holds_a_collapsing_policy_at_the_floor() -> None:
    """A softmax policy over 16 candidates, of which one alone is rewarded, collapses under the score-function loss
    unless the entropy weight rises."""
    torch.manual_seed(0)
    logits = torch.zeros(16, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=1e-2)
    objective = Reinforce(entropy_weight_rate=0.1)
    learner = objective.learner()
    entropies = []
    for _ in range(3000):
        scores = logits.expand(64, -1)
        taken = objective.explore(scores.detach())
        reward = (taken == 0).float()
        loss = learner.loss(scores, taken, (reward - reward.mean()) / reward.std().clamp_min(1e-6))
        learner.update(scores.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_probs = logits.detach().log_softmax(dim=0)
        entropies.append(-(log_probs.exp() * log_probs).sum().item())
    late = torch.tensor(entropies[-1000:])
    assert 0.7 < late.mean() < 1.5 and late.min() > 0.2, (late.mean(), late.min())
