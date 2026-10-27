"""ADE20K semantic segmentation with linear probes: CanViT after every glimpse of a paper policy or a learned
policy, and the passive DINOv3 baseline.

All save {"mious": {"t0": ..., "t1": ...}, "metadata": ...}: dataset-level mIoU over the
validation set, after each glimpse for CanViT and after its single forward pass for DINOv3.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from canvit_pytorch.benchmarks import ade20k
from canvit_pytorch.episode import EpisodeStep
from canvit_pytorch.evaluate.config import EpisodeConfig
from canvit_pytorch.evaluate.tasks.task import Task
from canvit_pytorch.hub import repos
from canvit_pytorch.hub.loading import checkpoint_record
from canvit_pytorch.model import CanViTForSemanticSegmentation
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.rl import LearnedPolicy
from canvit_pytorch.teacher import DINOV3_REPOS, DINOv3Variant, load_teacher

log = logging.getLogger(__name__)

UPSAMPLING_CHUNK = 8
"""Scenes upsampled at once: 150 classes of float32 logits at 512² px take 157 MB per scene."""


def predicted_labels(logits: Tensor, *, size_px: int) -> Tensor:
    """[B, num_classes, g, g] logits -> [B, size_px, size_px] labels: bilinear upsampling, then argmax."""
    return torch.cat([
        F.interpolate(chunk, size=(size_px, size_px), mode="bilinear", align_corners=False).argmax(dim=1)
        for chunk in logits.split(UPSAMPLING_CHUNK)
    ])


@dataclass(frozen=True, kw_only=True)
class ADE20kSegmentationCanViT(Task):
    """ADE20K mIoU after every glimpse: a linear probe decodes CanViT's canvas into a segmentation."""

    probe_repo: str
    """Hub repo or local directory of an ADE20K probe trained on this checkpoint's canvas at this canvas grid size."""
    pretrained_repo: str = repos.FLAGSHIP
    episode: EpisodeConfig = EpisodeConfig()
    scene_size_px: int = repos.RELEASED_SCENE_SIZE_PX
    """Side of the square scenes and label maps are resized to; glimpses are sampled from the scene."""
    output: Path = Path("results/ade20k_seg.pt")
    batch_size: int = 32

    @torch.inference_mode()
    def run(self) -> Path:
        seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(
            pretrained_repo=self.pretrained_repo, probe_repo=self.probe_repo,
        ).to(self.torch_device).eval()
        return save_mious_by_glimpse(
            self, seg, lambda images: self.episode.rollout(canvit=seg.canvit, images=images, canvas_logits=seg.logits),
            num_glimpses=self.episode.num_glimpses, scene_size_px=self.scene_size_px,
        )


@dataclass(frozen=True, kw_only=True)
class ADE20kSegmentationLearnedPolicy(Task):
    """ADE20K mIoU after every glimpse of a learned policy (canvit_pytorch.rl), with the observer its scorer records."""

    scorer_repo: str
    """Hub repo or local directory of a ViewpointScorer, such as one of hub.repos.Q_LEARNING_POLICIES."""
    scorer_revision: str | None = None
    """The Hub commit, branch or tag of scorer_repo to evaluate, the latest when None; the result records the commit."""
    num_glimpses: int
    """The full scene, then the policy's choices."""
    output: Path = Path("results/ade20k_seg_learned_policy.pt")
    batch_size: int = 8

    @torch.inference_mode()
    def run(self) -> Path:
        scorer = checkpoint_record(self.scorer_repo, revision=self.scorer_revision)
        policy = LearnedPolicy.from_pretrained(self.scorer_repo, self.torch_device, revision=scorer["revision"])
        return save_mious_by_glimpse(
            self, policy.model, lambda images: policy.rollout(images, num_glimpses=self.num_glimpses),
            num_glimpses=self.num_glimpses, scene_size_px=policy.scorer.observer.scene_size_px, scorer=scorer,
        )


def save_mious_by_glimpse(
    task: Task, seg: CanViTForSemanticSegmentation, rollout: Callable[[Tensor], list[EpisodeStep]], *,
    num_glimpses: int, scene_size_px: int, **metadata_entries: object,
) -> Path:
    """Run rollout on each batch of validation scenes [B, 3, S, S]; save the mIoU of seg's probe after every glimpse,
    with metadata_entries."""
    device = task.torch_device
    dataset = ade20k.validation_set(size_px=scene_size_px)
    mious = [ade20k.MeanIoU(device=device) for _ in range(num_glimpses)]
    start = time.monotonic()
    with task.autocast():
        for images, labels in task.batches(dataset, description="ADE20K"):
            steps = rollout(images.to(device))
            labels = labels.to(device)
            for step in steps:
                mious[step.t].update(predicted_labels(seg.logits(step.state.canvas), size_px=scene_size_px), labels)
    results = {f"t{t}": miou.compute() for t, miou in enumerate(mious)}
    log.info("mIoU by glimpse: %s", ", ".join(f"{100 * v:.2f}" for v in results.values()))
    return task.save(
        {"mious": results}, n_images=len(dataset), wall_time_seconds=time.monotonic() - start, **metadata_entries,
    )


@dataclass(frozen=True, kw_only=True)
class ADE20kSegmentationDINOv3(Task):
    """ADE20K mIoU of a frozen DINOv3 whose patch features a linear probe decodes: one passive forward pass."""

    input_size_px: int
    """DINOv3's input resolution, which its released probe was trained at; the scene is resized to it."""
    variant: DINOv3Variant = "vitb16"
    scene_size_px: int = repos.RELEASED_SCENE_SIZE_PX
    """Resolution of the label maps the prediction is upsampled to and scored against."""
    output: Path = Path("results/ade20k_seg_dinov3.pt")
    batch_size: int = 32

    @property
    def dinov3_repo(self) -> str:
        return DINOV3_REPOS[self.variant]

    @property
    def probe_repo(self) -> str:
        model_name = repos.DINOV3_PROBE_MODEL_NAMES[self.variant]
        return repos.released_dinov3_ade20k_probe(model_name, input_size_px=self.input_size_px)

    @torch.inference_mode()
    def run(self) -> Path:
        device = self.torch_device
        dataset = ade20k.validation_set(size_px=self.scene_size_px)
        teacher = load_teacher(self.dinov3_repo, device)
        probe = SegmentationProbe.from_pretrained(self.probe_repo).to(device).eval()
        miou = ade20k.MeanIoU(device=device)
        start = time.monotonic()
        with self.autocast():
            for images, labels in self.batches(dataset, description="ADE20K"):
                features = teacher.patch_grid(images.to(device), input_size_px=self.input_size_px)
                predictions = predicted_labels(probe(features), size_px=self.scene_size_px)
                miou.update(predictions, labels.to(device))
        result = miou.compute()
        log.info("mIoU: %.2f", 100 * result)
        return self.save(
            {"mious": {"t0": result}},
            n_images=len(dataset), dinov3_repo=self.dinov3_repo, probe_repo=self.probe_repo,
            wall_time_seconds=time.monotonic() - start,
        )
