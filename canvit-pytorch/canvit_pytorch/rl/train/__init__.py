"""Training a ViewpointScorer for a frozen observer: python -m canvit_pytorch.rl.train.

A training step runs an episode on each scene of a batch (rl.train.exploration): the full scene, then the
viewpoints the objective chooses from the scorer's scores. A glimpse's reward is the fraction of the probe's
cross-entropy it removes (rl.train.reward), standardized by running statistics kept per decision, and the
objective's loss fits the scores of the viewpoints taken. Every validation_interval steps, the deployed policy runs
on the validation scenes (rl.train.validation), and the run keeps the scorer of lowest mean cross-entropy after
the glimpses it chose.
"""

import dataclasses
import itertools
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from canvit_pytorch.benchmarks.ade20k import ADE20kDataset, dataset_root, evaluation_transform
from canvit_pytorch.benchmarks.ade20k.dataset import Split
from canvit_pytorch.progress import progress
from canvit_pytorch.provenance import provenance
from canvit_pytorch.rl.candidates import CandidateViewpoints
from canvit_pytorch.rl.features import FEATURE_GROUPS, FeatureGroup
from canvit_pytorch.rl.observer import Observer, flagship_observer
from canvit_pytorch.rl.policy import LearnedPolicy
from canvit_pytorch.rl.scorer import ViewpointScorer
from canvit_pytorch.rl.train.exploration import explore
from canvit_pytorch.rl.train.objectives import Objective, QLearning
from canvit_pytorch.rl.train.reward import RunningStandardizer
from canvit_pytorch.rl.train.validation import cross_entropy_by_glimpse

log = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class PolicyTraining:
    """Train a learned viewing policy on ADE20K. The defaults are the recipe of hub.repos.Q_LEARNING_POLICIES,
    one run per seed; with the Reinforce objective's defaults, that of hub.repos.REINFORCE_POLICIES."""

    out_dir: Path
    """Created by the run: the selected scorer (scorer/) and that of the last step (last_step_scorer/), both for
    ViewpointScorer.from_pretrained, and record.json."""
    objective: Objective = QLearning()
    seed: int = 0
    observer: Observer = flagship_observer(canvas_grid_size=64)
    feature_groups: tuple[FeatureGroup, ...] = FEATURE_GROUPS
    """The scorer's input (rl.features)."""
    candidates: CandidateViewpoints = CandidateViewpoints(scales=(0.5, 0.25), centers_per_axis=16)
    width: int = 128
    """Channels of the scorer's U-Net."""
    blocks_per_stage: int = 3
    """ConvNeXt-V2 blocks per stage of the U-Net and of the score head."""
    num_glimpses: int = 5
    """Glimpses per training and validation episode: the full scene, then the scorer's choices."""
    batch_size: int = 16
    glimpse_budget: int = 640_000
    """Glimpses of training episodes: glimpse_budget // (batch_size × num_glimpses) steps."""
    lr: float = 2e-4
    warmup_fraction: float = 0.125
    """The fraction of the steps over which the learning rate rises linearly from 0; it then stays at lr."""
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0
    reward_resolution_px: int = 128
    """The reward's cross-entropy is computed at this resolution; validation's at the labels' own."""
    reward_momentum: float = 0.997
    """Momentum of the running statistics that standardize the rewards of each decision."""
    validation_interval: int = 1000
    device: str = "cuda"
    amp: bool = True
    """Run CanViT under bfloat16 autocast; the probe, the features and the scorer compute in float32."""
    num_workers: int = 8

    def __post_init__(self) -> None:
        assert self.num_glimpses >= 2, f"an episode of {self.num_glimpses} glimpses has no decision to learn from"

    def scenes(self, split: Split) -> Dataset[tuple[Tensor, Tensor]]:
        """Images and label maps, resized to the observer's scene size without cropping; override to train on other
        data."""
        transform = evaluation_transform(self.observer.scene_size_px)
        return ADE20kDataset(root=dataset_root(), split=split, transform=transform)

    def loader(self, split: Split, device: torch.device) -> DataLoader[tuple[Tensor, Tensor]]:
        training = split == "training"
        return DataLoader(
            self.scenes(split), batch_size=self.batch_size, shuffle=training, drop_last=training,
            num_workers=self.num_workers, pin_memory=device.type == "cuda", persistent_workers=self.num_workers > 0,
        )

    def run(self) -> Path:
        """Train; return the directory of the selected scorer."""
        device = torch.device(self.device)
        torch.manual_seed(self.seed)
        model = self.observer.load(device)
        scorer = ViewpointScorer(
            observer=self.observer, canvas_dim=model.canvit.canvas_dim, feature_groups=self.feature_groups,
            candidates=self.candidates, width=self.width, blocks_per_stage=self.blocks_per_stage,
            dueling=self.objective.dueling,
        ).to(device)
        policy = LearnedPolicy(scorer=scorer, model=model)
        learner = self.objective.learner()
        standardize = RunningStandardizer(
            num_streams=self.num_glimpses - 1, momentum=self.reward_momentum, device=device,
        )
        num_steps = self.glimpse_budget // (self.batch_size * self.num_glimpses)
        warmup_steps = max(1, int(self.warmup_fraction * num_steps))
        optimizer = torch.optim.AdamW(
            scorer.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=self.adam_betas,
        )
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / warmup_steps))
        training_batches = itertools.chain.from_iterable(itertools.repeat(self.loader("training", device)))
        validation_loader = self.loader("validation", device)
        self.out_dir.mkdir(parents=True)
        record = {
            "objective": type(self.objective).__name__, "config": dataclasses.asdict(self),
            "provenance": provenance(device), "validations": [],
        }
        best = math.inf
        loss_sum, num_losses = torch.zeros((), device=device), 0
        log.info("%d steps of %d scenes; selected scorer and record in %s", num_steps, self.batch_size, self.out_dir)
        for step in progress(range(1, num_steps + 1), description="Training steps"):
            images, labels = (x.to(device, non_blocking=True) for x in next(training_batches))
            scorer.eval()
            decisions = explore(
                model, scorer, images, labels, choose=self.objective.explore, num_glimpses=self.num_glimpses,
                reward_resolution_px=self.reward_resolution_px, amp=self.amp,
            )
            scorer.train()
            scores = scorer(decisions.features)
            loss = learner.loss(scores, decisions.taken, standardize(decisions.rewards).flatten())
            learner.update(scores.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), self.max_grad_norm)
            optimizer.step()
            schedule.step()
            loss_sum += loss.detach()
            num_losses += 1
            if step % self.validation_interval and step != num_steps:
                continue
            training_loss = (loss_sum / num_losses).item()
            assert math.isfinite(training_loss), f"step {step}: training loss {training_loss}"
            loss_sum.zero_()
            num_losses = 0
            scorer.eval()
            cross_entropy = cross_entropy_by_glimpse(
                policy, validation_loader, num_glimpses=self.num_glimpses, device=device, amp=self.amp,
            )
            scorer.train()
            selection_value = sum(cross_entropy[1:]) / (self.num_glimpses - 1)
            assert math.isfinite(selection_value), f"step {step}: validation cross-entropy {cross_entropy}"
            selected = selection_value < best
            if selected:
                best = selection_value
                scorer.save_pretrained(self.out_dir / "scorer")
                record["selected_step"] = step
            record["validations"].append(
                {"step": step, "training_loss": training_loss, "validation_cross_entropy": cross_entropy},
            )
            (self.out_dir / "record.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
            log.info("step %d: training loss %.4f; validation cross-entropy by glimpse %s, mean after the chosen "
                     "glimpses %.4f%s", step, training_loss, " ".join(f"{v:.4f}" for v in cross_entropy),
                     selection_value, " (selected)" if selected else "")
        scorer.save_pretrained(self.out_dir / "last_step_scorer")
        return self.out_dir / "scorer"
