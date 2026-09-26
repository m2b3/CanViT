"""The probe-training loop, the same for every frozen feature source (paper, Appendix D.3).

One probe decodes every feature map of a scene (the canvas after each glimpse, or DINOv3's patch features);
the loss averages its cross-entropy over the maps.
"""

import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, assert_never

import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dinov3.eval.segmentation.schedulers import WarmupOneCycleLR
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from canvit_pytorch.benchmarks.ade20k import IGNORE_LABEL, NUM_CLASSES, MeanIoU
from canvit_pytorch.benchmarks.ade20k.dataset import Split
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.provenance import provenance
from canvit_pytorch.specialize.ade20k import data
from canvit_pytorch.specialize.ade20k.config import ProbeTrainingConfig
from canvit_pytorch.specialize.ade20k.figures import prediction_figure
from canvit_pytorch.specialize.ade20k.record import RunDirectory

log = logging.getLogger(__name__)


class FeatureMaps(Protocol):
    """A frozen model's features of ImageNet-normalized scenes [B, 3, S, S]: [B, H, W, embed_dim] maps, in order."""

    layer_normalized: ClassVar[bool]
    """The model's outputs are already layer-normalized, so the probe skips its LayerNorm."""

    @property
    def embed_dim(self) -> int: ...

    def __call__(self, images: Tensor, split: Split) -> list[Tensor]: ...


class ProbeSetup(Protocol):
    """A training command's configuration dataclass: the protocol, and the frozen model whose features it decodes."""

    @property
    def training(self) -> ProbeTrainingConfig: ...

    @property
    def run_name(self) -> str: ...

    def load_feature_maps(self, device: torch.device) -> FeatureMaps: ...


@dataclass(frozen=True)
class LabeledMaps:
    images: Tensor  # [B, 3, S, S]
    labels: Tensor  # [B, S, S]
    maps: list[Tensor]  # [B, H, W, D] each


def resize_labels(labels: Tensor, size: torch.Size) -> Tensor:
    """Nearest-neighbor resize of [B, H, W] class indices to [B, *size]."""
    if labels.shape[1:] == size:
        return labels
    return F.interpolate(labels.unsqueeze(1).float(), size=tuple(size), mode="nearest").squeeze(1).long()


def segmentation_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Cross-entropy of [B, C, H, W] logits against the labels resized to H×W."""
    return F.cross_entropy(logits, resize_labels(labels, logits.shape[2:]), ignore_index=IGNORE_LABEL)


def predict(probe: SegmentationProbe, features: Tensor, labels: Tensor) -> Tensor:
    """The predicted class of each pixel of the labels [B, S, S]: the probe's argmax, resized to the labels."""
    return resize_labels(probe(features.float()).argmax(1), labels.shape[1:])


def feature_autocast(config: ProbeTrainingConfig, device: torch.device) -> torch.autocast:
    match config.feature_dtype:
        case "bfloat16":
            return torch.autocast(device.type, dtype=torch.bfloat16)
        case "float32":
            return torch.autocast(device.type, enabled=False)
        case _:
            assert_never(config.feature_dtype)


class TrainingWindow:
    """Training loss, gradient norm and mIoU per feature map, accumulated on the device between two logs."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.loss_sum = torch.zeros((), device=self.device)
        self.grad_norm_sum = torch.zeros((), device=self.device)
        self.num_steps = 0
        self.ious: list[MeanIoU] = []

    def add(self, *, loss: Tensor, grad_norm: Tensor, predictions: list[Tensor], labels: Tensor) -> None:
        self.loss_sum += loss.detach()
        self.grad_norm_sum += grad_norm.detach()
        self.num_steps += 1
        self.ious = self.ious or [MeanIoU(device=self.device) for _ in predictions]
        for iou, prediction in zip(self.ious, predictions, strict=True):
            iou.update(prediction, labels)

    def log_and_reset(self, experiment: comet_ml.CometExperiment, *, step: int, lr: float, log_curve: bool) -> None:
        mious = [iou.compute() for iou in self.ious]
        experiment.log_metrics({
            "train/loss": (self.loss_sum / self.num_steps).item(),
            "train/grad_norm": (self.grad_norm_sum / self.num_steps).item(),
            "train/lr": lr,
            "train/miou_mean": sum(mious) / len(mious),
        }, step=step)
        if log_curve:
            experiment.log_curve("train/miou", x=list(range(len(mious))), y=mious, step=step)
        self.reset()


@torch.no_grad()
def validate(
    *, probe: SegmentationProbe, feature_maps: FeatureMaps, loader: DataLoader[tuple[Tensor, Tensor]],
    autocast: torch.autocast, device: torch.device, figure_samples: int,
) -> tuple[list[float], LabeledMaps]:
    """mIoU on each feature map over the validation set, and the first figure_samples scenes (copied, so the
    batch they come from can be freed)."""
    probe.eval()
    ious: list[MeanIoU] = []
    first_batch: LabeledMaps | None = None
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast:
            maps = feature_maps(images, "validation")
        ious = ious or [MeanIoU(device=device) for _ in maps]
        for iou, features in zip(ious, maps, strict=True):
            iou.update(predict(probe, features, labels), labels)
        first_batch = first_batch or LabeledMaps(
            images=images[:figure_samples].clone(), labels=labels[:figure_samples].clone(),
            maps=[features[:figure_samples].clone() for features in maps],
        )
    assert first_batch is not None, "the validation set is empty"
    return [iou.compute() for iou in ious], first_batch


def log_figure(
    experiment: comet_ml.CometExperiment, *, name: str, step: int, probe: SegmentationProbe, batch: LabeledMaps,
    num_samples: int,
) -> None:
    probe.eval()
    with torch.no_grad():
        labels, maps = batch.labels[:num_samples], [features[:num_samples] for features in batch.maps]
        predictions = [predict(probe, features, labels) for features in maps]
    figure = prediction_figure(images=batch.images[:num_samples], labels=labels, maps=maps, predictions=predictions)
    experiment.log_figure(figure_name=f"{name}_{step}", figure=figure, step=step)
    plt.close(figure)


def train_probe(setup: ProbeSetup) -> Path:
    """Train a probe as setup.training specifies; returns the run directory."""
    assert dataclasses.is_dataclass(setup) and not isinstance(setup, type), type(setup)
    config = setup.training
    torch.set_float32_matmul_precision("high")
    device = torch.device(config.device)
    feature_maps = setup.load_feature_maps(device)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    probe = SegmentationProbe(
        embed_dim=feature_maps.embed_dim, num_classes=NUM_CLASSES, dropout=config.dropout,
        use_ln=not feature_maps.layer_normalized,
    ).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.peak_lr, weight_decay=config.weight_decay)
    scheduler = WarmupOneCycleLR(
        optimizer, max_lr=config.peak_lr, total_steps=config.num_steps, warmup_iters=config.warmup_steps,
        warmup_ratio=config.warmup_start_lr_fraction, pct_start=0, anneal_strategy="cos",
        final_div_factor=float("inf"), use_beta1=False, update_momentum=False,
    )
    training_batches = data.training_batches(config)
    validation_loader = data.validation_loader(config)

    experiment = comet_ml.start(
        project_name=config.comet_project,
        experiment_config=comet_ml.ExperimentConfig(name=setup.run_name, auto_metric_logging=False),
    )
    experiment.log_parameters(dataclasses.asdict(setup))
    run = RunDirectory(
        config.output_dir / f"{setup.run_name}_{time.strftime('%Y%m%d-%H%M%S')}_{experiment.get_key()[:8]}",
        config_type=type(setup).__name__, config=dataclasses.asdict(setup),
        comet_experiment_key=experiment.get_key(), provenance=provenance(device),
    )
    log.info("Comet experiment %s/%s/%s; run directory %s; probe of %d parameters on %d-dim features",
             experiment.workspace, experiment.project_name, experiment.get_key(), run.path,
             sum(p.numel() for p in probe.parameters()), feature_maps.embed_dim)

    autocast = feature_autocast(config, device)

    def validate_and_record(step: int) -> LabeledMaps:
        start = time.perf_counter()
        mious, first_batch = validate(
            probe=probe, feature_maps=feature_maps, loader=validation_loader, autocast=autocast, device=device,
            figure_samples=config.figure_samples,
        )
        selected = run.record_validation(step=step, validation_miou=mious, probe=probe)
        for t, miou in enumerate(mious):
            experiment.log_metric(f"val/miou_t{t}", miou, step=step)
        experiment.log_curve("val/miou", x=list(range(len(mious))), y=mious, step=step)
        experiment.log_metric("timing/validation_seconds", time.perf_counter() - start, step=step)
        log.info("Step %d: validation mIoU per feature map [%s]; selected probe: step %d",
                 step, " ".join(f"{miou:.4f}" for miou in mious), selected.selected_step)
        return first_batch

    window = TrainingWindow(device)
    validation_batch: LabeledMaps | None = None
    for step in tqdm(range(config.num_steps), desc=setup.run_name):
        images, labels = (tensor.to(device) for tensor in next(training_batches))
        if step % config.validate_every == 0:
            validation_batch = validate_and_record(step)

        probe.train()
        with autocast:
            maps = feature_maps(images, "training")
        optimizer.zero_grad()
        logits = [probe(features.float()) for features in maps]
        loss = torch.stack([segmentation_loss(map_logits, labels) for map_logits in logits]).mean()
        loss.backward()
        grad_norm = torch.nn.utils.get_total_norm([p.grad for p in probe.parameters() if p.grad is not None])
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            predictions = [resize_labels(map_logits.argmax(1), labels.shape[1:]) for map_logits in logits]
        window.add(loss=loss, grad_norm=grad_norm, predictions=predictions, labels=labels)

        if step % config.figure_every == 0:
            start = time.perf_counter()
            assert validation_batch is not None
            for name, batch in (("train", LabeledMaps(images=images, labels=labels, maps=maps)), ("val", validation_batch)):
                log_figure(experiment, name=name, step=step, probe=probe, batch=batch, num_samples=config.figure_samples)
            experiment.log_metric("timing/figure_seconds", time.perf_counter() - start, step=step)

        steps_taken = step + 1
        if steps_taken % config.log_every == 0:
            window.log_and_reset(
                experiment, step=steps_taken, lr=float(scheduler.get_last_lr()[0]),
                log_curve=steps_taken % config.validate_every == 0,
            )

    validate_and_record(config.num_steps)
    experiment.end()
    return run.path
