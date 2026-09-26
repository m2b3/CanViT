"""ImageNet-1k classification from the recurrent CLS token after every glimpse.

Saves {"top_k_preds": int16 [N, T, 5], "labels": int16 [N], "metadata": ...} over the N
validation images and T glimpses, so any top-k accuracy up to 5 can be computed afterwards.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import torch
import tyro

from canvit_pytorch.benchmarks import imagenet
from canvit_pytorch.evaluate.config import SCENE_SIZE_PX, EpisodeConfig
from canvit_pytorch.evaluate.tasks.task import Task
from canvit_pytorch.hub import repos
from canvit_pytorch.model import CanViTForImageClassification
from canvit_pytorch.policies import POLICIES

log = logging.getLogger(__name__)

TOP_K = 5


@dataclass(frozen=True)
class FrozenWithFusedProbe:
    """A frozen pretrained CanViT whose CLS readout is fused with a linear probe fitted on its teacher's CLS token."""

    pretrained_repo: str = repos.FLAGSHIP
    probe_repo: str = repos.DINOV3_VITB16_IN1K_PROBE

    def load(self) -> CanViTForImageClassification:
        return CanViTForImageClassification.from_pretrained_with_probe(
            pretrained_repo=self.pretrained_repo, probe_repo=self.probe_repo,
        )


@dataclass(frozen=True)
class FineTuned:
    """CanViT fine-tuned end to end on ImageNet-1k classification."""

    repo: str = repos.FINETUNED_IN1K

    def load(self) -> CanViTForImageClassification:
        return CanViTForImageClassification.from_pretrained(self.repo)


Classifier = (
    Annotated[FrozenWithFusedProbe, tyro.conf.subcommand("frozen")]
    | Annotated[FineTuned, tyro.conf.subcommand("finetuned")]
)


@dataclass(frozen=True, kw_only=True)
class ImageNetClassification(Task):
    """ImageNet-1k top-5 predictions from the recurrent CLS token after every glimpse."""

    classifier: Classifier = FineTuned()
    episode: EpisodeConfig = EpisodeConfig()
    scene_size_px: int = SCENE_SIZE_PX
    """Images are resized on the short side and center-cropped to a square scene of this side."""
    output: Path = Path("results/in1k_clf.pt")
    batch_size: int = 64

    def __post_init__(self) -> None:
        spec = POLICIES[self.episode.policy]
        assert not spec.needs_segmentation_probe, f"{spec.paper_name} decodes the canvas with an ADE20K probe"

    @torch.inference_mode()
    def run(self) -> Path:
        device = self.torch_device
        dataset = imagenet.validation_set(imagenet.validation_dir(), size_px=self.scene_size_px)
        classifier = self.classifier.load().to(device).eval()
        num_images, num_glimpses = len(dataset), self.episode.num_glimpses
        top_k_preds = torch.zeros(num_images, num_glimpses, TOP_K, dtype=torch.int16)
        all_labels = torch.zeros(num_images, dtype=torch.int16)
        correct_top1 = torch.zeros(num_glimpses, dtype=torch.long, device=device)
        start, done = time.monotonic(), 0
        with self.autocast():
            for images, labels in self.batches(dataset, description="ImageNet-1k"):
                steps = self.episode.rollout(canvit=classifier.canvit, images=images.to(device))
                batch = slice(done, done + labels.shape[0])
                all_labels[batch] = labels.to(torch.int16)
                labels = labels.to(device)
                for step in steps:
                    top_k = classifier.classify(step.state.recurrent_cls).topk(TOP_K, dim=-1).indices
                    top_k_preds[batch, step.t] = top_k.cpu().to(torch.int16)
                    correct_top1[step.t] += (top_k[:, 0] == labels).sum()
                done = batch.stop
        assert done == num_images, (done, num_images)
        log.info("Top-1 by glimpse: %s", ", ".join(f"{100 * n / num_images:.2f}" for n in correct_top1.tolist()))
        return self.save(
            {"top_k_preds": top_k_preds, "labels": all_labels},
            classifier=type(self.classifier).__name__, n_images=num_images, wall_time_seconds=time.monotonic() - start,
        )
