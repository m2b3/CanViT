"""ADE20K semantic segmentation evaluation.

Two entry points, matching the two model families we compare:

    CanViTConfig.run()   → multi-timestep CanViT episode, mIoU per timestep.
    DINOv3Config.run()   → single passive forward pass of a DINOv3 teacher, mIoU at t=0.

Both save .pt files with the same schema (`mious: {t0: ..., ...}` + metadata)
so the downstream export pipeline is model-agnostic.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from canvit_pytorch import CanViTForSemanticSegmentation, SegmentationProbe
from canvit_pytorch.probes import SegmentationProbe as SegmentationProbeType
from canvit_pytorch.teacher import load_teacher
from canvit_specialize.datasets.ade20k import (
    IGNORE_LABEL, NUM_CLASSES, ADE20kDataset, ResizeMode, make_val_transforms,
)
from canvit_specialize.metrics import mIoUAccumulator
from canvit_specialize.training.utils import collect_metadata
from torch import Tensor
from torch.utils.data import DataLoader

from canvit_eval.config import EpisodeConfig, TEACHER_REPO, ade20k_root, progress, require_existing_dir
from canvit_eval.runner import eval_batches
from canvit_eval.tasks.base import TaskConfig

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ADE20kBaseConfig(TaskConfig):
    """ADE20K-specific fields shared by both model paths."""
    probe_repo: str
    ade20k_root: Path = field(default_factory=ade20k_root)
    scene_size: int = 512
    resize_mode: ResizeMode = "squish"


@dataclass(kw_only=True)
class CanViTConfig(ADE20kBaseConfig):
    """ADE20K seg with a CanViT model — multi-timestep episode."""
    output: Path = Path("results/ade20k_seg.pt")
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)

    def run(self) -> Path:
        return run_canvit(self)


@dataclass(kw_only=True)
class DINOv3Config(ADE20kBaseConfig):
    """ADE20K seg with a DINOv3 backbone — single passive forward pass."""
    output: Path = Path("results/ade20k_seg.pt")
    eval_resolution: int
    """Resolution the probe was trained at (e.g. 128, 192, 512). Teacher runs at this size,
    NOT at scene_size. Required — a mismatch silently degrades mIoU so there's no default."""
    teacher_repo: str = TEACHER_REPO

    def run(self) -> Path:
        return run_dinov3(self)


def _loader(cfg: ADE20kBaseConfig) -> DataLoader:
    require_existing_dir(cfg.ade20k_root, description="ADE20K root", env_var="ADE20K_ROOT")
    img_tf, mask_tf = make_val_transforms(cfg.scene_size, cfg.resize_mode)
    dataset = ADE20kDataset(
        root=cfg.ade20k_root, split="validation",
        img_transform=img_tf, mask_transform=mask_tf,
    )
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )


def _update_miou(
    acc: mIoUAccumulator, probe: SegmentationProbeType, features: Tensor, masks: Tensor,
) -> None:
    """Run probe on features, interpolate to mask size if needed, accumulate IoU.
    Argmax + mask stay on GPU; `acc.update` itself avoids syncs."""
    with torch.autocast(device_type=features.device.type, enabled=False):
        logits = probe(features.float())
    if logits.shape[-1] != masks.shape[-1]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    acc.update(logits.argmax(dim=1), masks)


def _save(output: Path, mious: dict[str, float], cfg: object, extra_meta: dict, wall_time: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mious": mious,
        "metadata": {**collect_metadata(cfg), **extra_meta, "wall_time_seconds": wall_time},
    }, output)
    for k, v in mious.items():
        log.info("  %s: %.2f%%", k, 100 * v)
    log.info("Saved to %s (%.1fs)", output, wall_time)


@torch.inference_mode()
def run_canvit(cfg: CanViTConfig) -> Path:
    """CanViT ADE20K: T-step episode, mIoU accumulator per timestep.

    `torch.inference_mode` is REQUIRED — without it the T=21 state propagation
    builds an autograd tape that grows per-step and OOMs on any canvas grid.

    Per-batch work stays on GPU — `acc.update(argmax, masks)` adds to the
    accumulator without a sync. Host syncs only happen in the final
    `acc.compute()` loop after the full dataset is processed.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(cfg.device)
    require_existing_dir(cfg.ade20k_root, description="ADE20K root", env_var="ADE20K_ROOT")

    seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(
        pretrained_repo=cfg.episode.model_repo,
        probe_repo=cfg.probe_repo,
    ).to(device).eval()
    probe = seg.head

    canvas_grid = cfg.episode.canvas_grid or cfg.scene_size // seg.canvit.backbone.patch_size_px
    loader = _loader(cfg)
    T = cfg.episode.n_timesteps

    accs = [mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(T)]
    t_start = time.monotonic()
    n_images = 0
    policy_kwargs = {"probe": probe, "get_spatial_fn": seg.canvit.get_spatial} \
                    if cfg.episode.policy == "entropy_coarse_to_fine" else {}

    for br in eval_batches(
        model=seg.canvit, loader=loader, episode_cfg=cfg.episode,
        canvas_grid=canvas_grid, device=device, amp=cfg.amp,
        policy_kwargs=policy_kwargs,
    ):
        _, masks = br.batch
        masks = masks.to(device, non_blocking=True)
        B = masks.shape[0]
        n_images += B
        for step in br.steps:
            spatial = seg.canvit.get_spatial(step.state.canvas).view(B, canvas_grid, canvas_grid, -1)
            _update_miou(accs[step.t], probe, spatial, masks)

    mious = {f"t{t}": acc.compute() for t, acc in enumerate(accs)}
    wall = time.monotonic() - t_start
    _save(cfg.output, mious, cfg, {
        "n_images": n_images,
        "canvas_grid": canvas_grid,
        "policy": cfg.episode.policy,
        "n_timesteps": T,
        "model_repo": cfg.episode.model_repo,
        "probe_repo": cfg.probe_repo,
    }, wall)
    return cfg.output


def run_dinov3(cfg: DINOv3Config) -> Path:
    """DINOv3 passive ADE20K: single forward pass at cfg.eval_resolution, mIoU at t=0."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(cfg.device)
    require_existing_dir(cfg.ade20k_root, description="ADE20K root", env_var="ADE20K_ROOT")

    teacher = load_teacher(cfg.teacher_repo, device)
    probe = SegmentationProbe.from_pretrained(cfg.probe_repo).to(device).eval()
    grid = cfg.eval_resolution // 16  # DINOv3 patch size
    loader = _loader(cfg)

    acc = mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
    t_start = time.monotonic()
    n_images = 0
    amp_dtype = torch.bfloat16 if cfg.amp else torch.float32

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp):
        for images, masks in progress(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            B = images.shape[0]
            n_images += B
            resized = F.interpolate(
                images, size=(cfg.eval_resolution, cfg.eval_resolution),
                mode="bilinear", align_corners=False,
            )
            feats = teacher.forward_norm_features(resized).patches.view(B, grid, grid, -1)
            _update_miou(acc, probe, feats, masks)

    mious = {"t0": acc.compute()}
    wall = time.monotonic() - t_start
    _save(cfg.output, mious, cfg, {
        "n_images": n_images,
        "eval_resolution": cfg.eval_resolution,
        "teacher_repo": cfg.teacher_repo,
        "probe_repo": cfg.probe_repo,
    }, wall)
    return cfg.output
