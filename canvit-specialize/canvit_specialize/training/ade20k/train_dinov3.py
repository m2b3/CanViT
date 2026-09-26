"""DINOv3 baseline probe training on ADE20K. One probe, one resolution, one forward pass per image."""

import logging
import os
import time
from dataclasses import asdict, dataclass

import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
from canvit_pytorch.teacher import DINOv3Teacher, load_teacher
from tqdm import tqdm

from canvit_specialize.datasets.ade20k import IGNORE_LABEL, NUM_CLASSES
from canvit_specialize.training.ade20k.eval_utils import eval_probe_on_batch
from canvit_pytorch.probes import SegmentationProbe
from canvit_specialize.training.ade20k.common import make_ade20k_loaders, make_amp_ctx, make_optimizer_and_scheduler
from canvit_specialize.training.ade20k.config import ProbeTrainBase
from canvit_specialize.training.ade20k.loss import ce_loss, upsample_preds
from canvit_specialize.training.ade20k.viz import log_probe_viz
from canvit_specialize.metrics import mIoUAccumulator

log = logging.getLogger(__name__)


@dataclass
class DINOv3ProbeTrainConfig(ProbeTrainBase):
    """DINOv3 baseline probe training configuration."""

    resolution: int = 128
    model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def _extract_features(teacher: DINOv3Teacher, images: torch.Tensor, resolution: int) -> torch.Tensor:
    """Resize images to resolution, forward through teacher, return spatial features [B, H, W, D]."""
    patch_size = teacher.model.config.patch_size
    # Snap resolution to multiple of patch_size
    grid = resolution // patch_size
    assert grid > 0, f"resolution={resolution} too small for patch_size={patch_size}"
    sz = grid * patch_size
    resized = F.interpolate(images, size=(sz, sz), mode="bilinear", align_corners=False)
    feats = teacher.forward_norm_features(resized).patches
    return feats.view(images.shape[0], grid, grid, -1)


def train(cfg: DINOv3ProbeTrainConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device)

    log.info("=" * 60)
    log.info("DINOv3 Baseline Probe Training (ADE20K)")
    log.info("=" * 60)
    log.info(f"Model: {cfg.model}")
    log.info(f"Resolution: {cfg.resolution}px")
    log.info(f"Training: BS={cfg.batch_size}, steps={cfg.max_steps}, LR={cfg.peak_lr}")

    # Teacher
    teacher = load_teacher(cfg.model, device)
    patch_size = teacher.model.config.patch_size
    grid = cfg.resolution // patch_size
    assert grid > 0, f"resolution={cfg.resolution} too small for patch_size={patch_size}"
    log.info(f"  embed_dim={teacher.embed_dim}, patch_size={patch_size}, grid={grid}x{grid}")

    # Probe (no LN needed — teacher features are already post-LN)
    probe = SegmentationProbe(embed_dim=teacher.embed_dim, num_classes=NUM_CLASSES, dropout=cfg.dropout, use_ln=False).to(device)
    optimizer, scheduler = make_optimizer_and_scheduler(
        probe.parameters(), lr=cfg.peak_lr, weight_decay=cfg.weight_decay,
        max_steps=cfg.max_steps, warmup_steps=cfg.warmup_steps, warmup_lr_ratio=cfg.warmup_lr_ratio,
    )
    log.info(f"  probe params: {sum(p.numel() for p in probe.parameters()):,}")

    # IoU
    val_iou = mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
    train_iou = mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
    best_miou = 0.0

    # Data
    train_loader, val_loader = make_ade20k_loaders(cfg)

    # Comet
    exp = comet_ml.Experiment()
    model_short = cfg.model.split("/")[-1].replace("-pretrain-lvd1689m", "").replace("-pretrain", "")
    exp_name = f"{model_short}_{cfg.resolution}px_{time.strftime('%Y-%m-%d-%H%M%S-%Z')}"
    exp.set_name(exp_name)
    exp.log_parameters(asdict(cfg))
    exp.add_tag("dinov3-baseline")
    exp.add_tag(model_short)
    metric_prefix = f"{model_short}_{cfg.resolution}px"
    log.info(f"Comet: {exp.workspace}/{exp.project_name}/{exp.get_key()} ({exp_name})")

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (
        cfg.probe_ckpt_dir / f"{model_short}_{cfg.resolution}px_{timestamp}_{job_id}_{exp.get_key()[:8]}"
        if cfg.probe_ckpt_dir else None
    )
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Checkpoints: {run_dir}")

    amp_ctx = make_amp_ctx(cfg.amp, device)

    log.info("=" * 60)
    log.info("Starting training...")

    step = 0
    train_iter = iter(train_loader)
    pbar = tqdm(total=cfg.max_steps, desc="Training")
    loss_acc: torch.Tensor | None = None
    grad_acc: torch.Tensor | None = None
    acc_count = 0

    while step < cfg.max_steps:
        try:
            images, masks = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, masks = next(train_iter)
        images, masks = images.to(device), masks.to(device)

        if step % cfg.val_every == 0:
            val_start = time.perf_counter()
            probe.eval()
            val_iou.reset()

            viz_feats = viz_images = viz_masks = None
            do_viz = step % cfg.viz_every == 0

            with torch.no_grad():
                for vi, vm in val_loader:
                    vi, vm = vi.to(device), vm.to(device)
                    with amp_ctx:
                        feats = _extract_features(teacher, vi, cfg.resolution)
                    eval_probe_on_batch(probe, feats, vm, val_iou)
                    if do_viz and viz_feats is None:
                        viz_feats, viz_images, viz_masks = feats, vi, vm

            if do_viz and viz_feats is not None:
                assert viz_images is not None and viz_masks is not None
                log_probe_viz(exp, step, probe, viz_feats, viz_images, viz_masks, cfg.viz_samples)

            miou = val_iou.compute()
            improved = miou > best_miou
            if improved:
                best_miou = miou
            exp.log_metric(f"{metric_prefix}/val_miou", miou, step=step)
            exp.log_metric(f"{metric_prefix}/best_val_miou", best_miou, step=step)
            log.info(f"Step {step}: val mIoU={100*miou:.2f}% (best={100*best_miou:.2f}%)")

            if improved and run_dir:
                path = run_dir / f"best_miou{miou:.4f}_step{step}.pt"
                for old in run_dir.glob("best_*.pt"):
                    old.unlink()
                torch.save({
                    "probe_state_dict": probe.state_dict(),
                    "resolution": cfg.resolution,
                    "model": cfg.model,
                    "best_miou": best_miou,
                    "step": step,
                    "config": asdict(cfg),
                }, path)
                log.info(f"  Saved best: {path}")

            val_time = time.perf_counter() - val_start
            exp.log_metric("timing/val_seconds", val_time, step=step)

        probe.train()
        optimizer.zero_grad()

        with amp_ctx:
            feats = _extract_features(teacher, images, cfg.resolution)
        logits = probe(feats.float())
        loss = ce_loss(logits, masks)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(probe.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        loss_acc = loss.detach() if loss_acc is None else loss_acc + loss.detach()
        grad_acc = grad_norm.detach() if grad_acc is None else grad_acc + grad_norm.detach()
        acc_count += 1

        # Train IoU
        with torch.no_grad():
            preds = logits.detach().argmax(1)
            preds_up = upsample_preds(preds, masks.shape[1], masks.shape[2])
            train_iou.update(preds_up, masks)

        step += 1
        pbar.update(1)

        if step % cfg.log_every == 0:
            assert loss_acc is not None and grad_acc is not None
            avg_loss = (loss_acc / acc_count).item()
            avg_grad = (grad_acc / acc_count).item()
            train_miou = train_iou.compute()
            exp.log_metrics({
                f"{metric_prefix}/loss": avg_loss,
                f"{metric_prefix}/grad_norm": avg_grad,
                f"{metric_prefix}/lr": scheduler.get_last_lr()[0],
                f"{metric_prefix}/train_miou": train_miou,
            }, step=step)
            loss_acc = grad_acc = None
            acc_count = 0
            train_iou.reset()

    pbar.close()

    # Final checkpoint
    if run_dir:
        path = run_dir / f"final_step{step}.pt"
        torch.save({
            "probe_state_dict": probe.state_dict(),
            "resolution": cfg.resolution,
            "model": cfg.model,
            "best_miou": best_miou,
            "step": step,
            "config": asdict(cfg),
        }, path)
        log.info(f"Final checkpoint: {path}")

    log.info("=" * 60)
    log.info(f"Training complete. Best mIoU: {100*best_miou:.2f}%")
    exp.log_metric(f"{metric_prefix}/best_miou", best_miou)
    log.info("=" * 60)
