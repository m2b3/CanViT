"""Main training loop."""

# backward() runs outside autocast (as PyTorch recommends). torch.compile's default
# "same_as_forward" assumption silently corrupts gradients when that's the case.
import torch._functorch.config

torch._functorch.config.backward_pass_autocast = "off"  # type: ignore[attr-defined]

import logging
import os
import signal
import subprocess
import time
import traceback
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import comet_ml
import dacite
import numpy as np
import optuna
import torch
from torch import Tensor, nn
from tqdm import tqdm


class TrainBatch(NamedTuple):
    """A training batch with precomputed targets (viewpoints sampled separately)."""
    images: Tensor
    labels: Tensor  # ImageNet class labels (for probe-based accuracy, if probe available)
    scene_target: Tensor  # Normalized teacher scene features
    cls_target: Tensor  # Normalized teacher CLS features
    raw_scene_target: Tensor  # Raw teacher scene features (for metrics)
    raw_cls_target: Tensor  # Raw teacher CLS features (for metrics)

# Force FlashAttention for SDPA - fail loud if unavailable
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
import torch.nn.functional as F  # noqa: E402
from canvit_pytorch import (  # noqa: E402
    CanViT,
    CanViTConfig,
    CanViTForPretraining,
    CanViTForRGBReconstruction,
    CLSStandardizer,
    PatchStandardizer,
    Viewpoint,
    patchify,
    sample_at_viewpoint,
)
from canvit_pytorch.backbone.vit import NormFeatures  # noqa: E402

from canvit_pretrain import CanViTForPretrainingConfig  # noqa: E402
from canvit_pretrain.checkpoint import (  # noqa: E402
    CheckpointData,
    current_provenance,
    find_latest,
    load_state_dict_flexible,
    update_symlink,
)
from canvit_pretrain.checkpoint import load as load_checkpoint  # noqa: E402
from canvit_pretrain.checkpoint import save as save_checkpoint  # noqa: E402

from .config import Config  # noqa: E402
from .data import ShardedFeatureLoader, create_loaders, scene_size_px  # noqa: E402
from .ema import EMATracker  # noqa: E402
from .model import compile_model, compile_teacher, create_model, load_student_backbone, load_teacher  # noqa: E402
from .objective import (  # noqa: E402
    distillation_branch_metrics_fn,
    distillation_loss_fn,
    rgb_branch_metrics_fn,
    rgb_loss_fn,
)
from .probe import load_probe  # noqa: E402
from .scheduler import warmup_constant_scheduler, warmup_cosine_scheduler  # noqa: E402
from .step import LossFn, training_step  # noqa: E402
from .utils import count_parameters  # noqa: E402
from .viewpoint import Viewpoint as NamedViewpoint  # noqa: E402
from .viz import log_figure, plot_multistep_pca, plot_multistep_recon, validate  # noqa: E402
from .viz.image import imagenet_denormalize_to_numpy  # noqa: E402

log = logging.getLogger(__name__)

# Signal-triggered checkpoint
_checkpoint_requested = False


def _handle_sigusr1(signum: int, frame: object) -> None:
    global _checkpoint_requested
    _checkpoint_requested = True
    log.info("SIGUSR1 received - will save checkpoint after current step")


def grad_norms_by_module(model: nn.Module, depth: int = 1) -> dict[str, float]:
    """Gradient norms grouped by module path prefix."""
    groups: dict[str, list[Tensor]] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        parts = name.split(".")
        prefix = ".".join(parts[:depth])
        groups.setdefault(prefix, []).append(param.grad)
    return {
        prefix: torch.cat([g.flatten() for g in grads]).norm().item()
        for prefix, grads in groups.items()
    }


def init_normalizer_stats_from_shard(
    shard_path: Path,
    scene_norm: PatchStandardizer,
    cls_norm: CLSStandardizer,
    device: torch.device,
    max_samples: int,
) -> None:
    """Initialize normalizer stats from one precomputed shard.

    Uses mmap + subset to avoid loading the full shard into memory
    (shards may be tens of GB).
    """
    log.info(f"Computing normalizer stats from shard: {shard_path.name}")
    shard = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
    n_total = shard["patches"].shape[0]
    n = min(n_total, max_samples) if max_samples > 0 else n_total
    # .clone() materializes from mmap; .float().to(device) for set_stats
    patches = shard["patches"][:n].clone().float().to(device)  # [n, n_tokens, D]
    cls = shard["cls"][:n].clone().float().to(device)  # [n, D]
    scene_norm.set_stats(patches)
    cls_norm.set_stats(cls.unsqueeze(1))  # [n, 1, D] for n_tokens=1
    log.info(f"  Scene/CLS stats from {n}/{n_total} samples")
    del shard, patches, cls
    torch.cuda.empty_cache()


def rgb_validate(
    *,
    exp: comet_ml.CometExperiment,
    step: int,
    model: CanViTForRGBReconstruction,
    images: Tensor,
    canvas_grid_size: int,
    glimpse_size_px: int,
    n_steps: int,
    min_viewpoint_scale: float,
) -> None:
    """Validation for rgb_recon: pixel reconstruction MSE over a held-out rollout.

    Mirrors training (full-scene t0, then random glimpses) and logs recon MSE at t0
    and at the final timestep — the teacher-free analog of the distillation val metrics.
    """
    was_training = model.training
    model.eval()
    target = patchify(images, model.patch_px)
    B = images.shape[0]
    state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid_size)
    mse_t0: Tensor | None = None
    mse: Tensor | None = None
    with torch.inference_mode():
        for t in range(n_steps):
            named = NamedViewpoint.full_scene(batch_size=B, device=images.device) if t == 0 \
                else NamedViewpoint.random(batch_size=B, device=images.device, min_scale=min_viewpoint_scale)
            vp = Viewpoint(centers=named.centers, scales=named.scales)
            glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=glimpse_size_px)
            state = model(glimpse=glimpse, state=state, viewpoint=vp).state
            mse = F.mse_loss(model.predict_rgb_patches(state.canvas), target)
            if t == 0:
                mse_t0 = mse
        recon_img = model.predict_rgb_image(state.canvas)[0]  # [3, H, W] after the full rollout
    assert mse is not None and mse_t0 is not None
    exp.log_metrics({"val/recon_loss": mse.item(), "val/recon_loss_t0": mse_t0.item()}, step=step)

    # Reconstruction panel: target | reconstruction (sample 0), the natural viz for a pixel objective.
    target_np = imagenet_denormalize_to_numpy(images[0])
    recon_np = imagenet_denormalize_to_numpy(recon_img)
    panel = np.concatenate([target_np, recon_np], axis=1).clip(0.0, 1.0)
    exp.log_image((panel * 255).astype(np.uint8), name="val/reconstruction", step=step)

    if was_training:
        model.train()


def train(cfg: Config, trial: optuna.Trial) -> float:
    """Train with stochastic reset. Returns best val_loss."""
    signal.signal(signal.SIGUSR1, _handle_sigusr1)
    log.info(f"Starting trial {trial.number}")
    log.info(f"Device: {cfg.device}")

    # === RUN NAME AND CHECKPOINT RESOLUTION ===
    run_name = cfg.run_name
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = cfg.ckpt_dir / run_name
    log.info(f"Run: {run_name} (dir: {run_dir})")

    # Check for failure marker (prevents infinite crash loops in job arrays)
    failed_marker = run_dir / "FAILED"
    if failed_marker.exists():
        log.error(f"FAILED marker exists: {failed_marker}")
        log.error(f"Previous job crashed. Delete marker to retry: rm {failed_marker}")
        cancel_slurm_array()
        raise RuntimeError(f"Refusing to start: {failed_marker} exists")

    # Create run_dir early so we can write FAILED marker on crash
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        return training_loop(cfg=cfg, trial=trial, run_name=run_name, run_dir=run_dir)
    except Exception:
        log.exception("Training crashed - writing FAILED marker")
        failed_marker.write_text(f"Crashed at {datetime.now(UTC).isoformat()}\n")
        cancel_slurm_array()
        raise


def cancel_slurm_array() -> None:
    """Cancel remaining SLURM array tasks. No-op if not in a SLURM job."""
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    if job_id is None:
        return
    log.info(f"Cancelling SLURM array job {job_id}")
    try:
        subprocess.run(["scancel", job_id], check=True)
    except Exception:
        log.exception(f"Failed to cancel SLURM job {job_id}")


def training_loop(*, cfg: Config, trial: optuna.Trial, run_name: str, run_dir: Path) -> float:
    """Main training loop. Called by train() with crash handling wrapper."""
    from dataclasses import asdict

    def flatten_dict(d: dict, prefix: str = "") -> dict[str, object]:
        flat: dict[str, object] = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_dict(v, f"{key}."))
            else:
                flat[key] = str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
        return flat

    is_rgb = cfg.objective == "rgb_recon"

    # Determine checkpoint source and load mode
    # Priority: run_dir/latest.pt (RESUME) > seed_ckpt (SEED) > hf_seed_ckpt (HF SEED) > fresh
    assert not (cfg.seed_ckpt and cfg.hf_seed_ckpt), "seed_ckpt and hf_seed_ckpt are mutually exclusive"
    ckpt_path_to_load: Path | None = None
    hf_seed_state_dict: dict[str, Tensor] | None = None
    is_seeding = False  # True = seed mode (weights only), False = resume mode (full state)
    latest = find_latest(run_dir)
    if latest is not None:
        ckpt_path_to_load = latest
        is_seeding = False
        log.info(f"RESUME mode: continuing from {ckpt_path_to_load}")
    elif cfg.seed_ckpt is not None:
        ckpt_path_to_load = cfg.seed_ckpt
        is_seeding = True
        log.info(f"SEED mode: loading weights from {ckpt_path_to_load} (fresh opt/sched/step)")
    elif cfg.hf_seed_ckpt is not None:
        assert not is_rgb, "hf_seed_ckpt is distillation-only; rgb_recon starts fresh or resumes a local ckpt"
        from canvit_pytorch.model.pretraining.hub import CanViTForPretrainingHFHub
        log.info(f"HF SEED mode: loading from {cfg.hf_seed_ckpt}")
        hf_model = CanViTForPretrainingHFHub.from_pretrained(cfg.hf_seed_ckpt)
        hf_seed_state_dict = {k: v for k, v in hf_model.state_dict().items()}
        assert isinstance(hf_model.cfg, CanViTForPretrainingConfig)
        cfg.model = hf_model.cfg
        log.info(f"  Model config from HF: {cfg.model}")
        del hf_model
        is_seeding = True
    else:
        log.info("FRESH mode: no checkpoint, starting from scratch")

    # Load checkpoint BEFORE creating Comet experiment
    ckpt_data: CheckpointData | None = None
    prev_comet_id: str | None = None
    if ckpt_path_to_load is not None:
        ckpt_data = load_checkpoint(ckpt_path_to_load, cfg.device)
        prev_comet_id = ckpt_data["comet_id"]
        # Override cfg.model from checkpoint — model arch MUST match saved weights.
        # Without this, CLI defaults (e.g. convex) override the checkpoint's config
        # (e.g. additive), causing missing/unexpected keys on load_state_dict.
        # rgb_recon rebuilds its CanViTConfig from ckpt model_config at model creation.
        if not is_rgb:
            ckpt_model_cfg = dacite.from_dict(CanViTForPretrainingConfig, ckpt_data["model_config"])
            if ckpt_model_cfg != cfg.model:
                log.warning(f"Overriding cfg.model from checkpoint (was {cfg.model.canvas_update_mode}, "
                            f"now {ckpt_model_cfg.canvas_update_mode})")
                cfg.model = ckpt_model_cfg

    # === COMET EXPERIMENT ===
    # RESUME mode: continue existing experiment. SEED/FRESH mode: new experiment.
    comet_cfg = comet_ml.ExperimentConfig(auto_metric_logging=False)
    if prev_comet_id is not None and not is_seeding:
        log.info(f"Continuing Comet experiment: {prev_comet_id}")
        exp = comet_ml.start(
            experiment_key=prev_comet_id,
            experiment_config=comet_cfg,
        )
    else:
        if is_seeding and prev_comet_id:
            log.info(f"SEED mode: creating new experiment (seed source had {prev_comet_id})")
        else:
            log.info("Creating NEW Comet experiment")
        exp = comet_ml.start(experiment_config=comet_cfg)

    exp.log_parameters(flatten_dict(asdict(cfg)))
    exp.log_parameters({"trial_number": trial.number, "run_name": run_name})
    if slurm_job_id := os.environ.get("SLURM_JOB_ID"):
        exp.log_parameters({"slurm_job_id": slurm_job_id})

    # Teacher + IN1k probe exist only for the distillation objective. rgb_recon is teacher-free.
    teacher = None if is_rgb else load_teacher(cfg)
    probe = None if is_rgb else load_probe(cfg.teacher_name, cfg.device)
    if teacher is not None:
        log.info(f"Teacher params: {count_parameters(teacher):,}")
    if probe is not None:
        log.info(f"Loaded IN1k probe for {cfg.teacher_name}")

    student_backbone = load_student_backbone(cfg)
    log.info(f"Student backbone params: {count_parameters(student_backbone):,}")

    model: CanViT
    if is_rgb:
        # rgb model config: from checkpoint when resuming, else from cfg.model's base fields.
        if ckpt_data is not None and not is_seeding:
            rgb_cfg = dacite.from_dict(CanViTConfig, ckpt_data["model_config"])
        else:
            rgb_cfg = CanViTConfig(**{k: getattr(cfg.model, k) for k in CanViTConfig.__dataclass_fields__})
        model = CanViTForRGBReconstruction(
            backbone=student_backbone, cfg=rgb_cfg, backbone_name=cfg.backbone_name,
        ).to(cfg.device)
        glimpse_size_px = cfg.glimpse_grid_size * student_backbone.patch_size_px
        log.info(f"RGB-reconstruction model: glimpse={glimpse_size_px}px, patch={model.patch_px}px")
        if cfg.compile:
            log.info("Compiling model (rgb_recon, no teacher)")
            model.compile()
        patch_size = student_backbone.patch_size_px
    else:
        assert teacher is not None
        bundle = create_model(student_backbone, teacher.embed_dim, cfg)
        model, glimpse_size_px = bundle.model, bundle.glimpse_size_px
        if cfg.compile:
            if cfg.combo_kernels:
                torch._inductor.config.combo_kernels = True  # type: ignore[attr-defined]
                log.info("Compiling teacher and model (combo_kernels=True, backward_pass_autocast=off)")
            else:
                log.info("Compiling teacher and model (backward_pass_autocast=off)")
            compile_teacher(teacher)
            compile_model(model)
        patch_size = teacher.model.config.patch_size

    G = cfg.canvas_patch_grid_size
    scene_size = scene_size_px(G, patch_size)
    log.info(f"Grid size: {G}, scene size: {scene_size}px")

    # Extract start_step from checkpoint scheduler state (BEFORE creating loaders)
    # NOTE: PyTorch LRScheduler uses "last_epoch" but we call scheduler.step() once per
    # training step, so last_epoch == number of gradient updates == our "step"
    # SEED mode always starts at step=0 (fresh training run)
    if ckpt_data is not None and not is_seeding:
        sched_state = ckpt_data["scheduler_state"]
        assert sched_state is not None, "Checkpoint has no scheduler_state — cannot determine start_step"
        start_step = sched_state["last_epoch"]  # PyTorch API: last_epoch = num scheduler.step() calls
        log.info("=" * 60)
        log.info(f"RESUME: start_step={start_step}")
        log.info("=" * 60)
    else:
        start_step = 0
        log.info("=" * 60)
        log.info(f"{'SEED' if is_seeding else 'FRESH'}: start_step=0")
        log.info("=" * 60)

    train_loader, val_loader = create_loaders(cfg, start_step=start_step)

    # Feature-based training is the only supported path
    assert cfg.feature_base_dir is not None, "feature_base_dir required (raw image training removed)"
    assert isinstance(train_loader, ShardedFeatureLoader)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = count_parameters(model)
    log.info(f"Model total: {n_total:,}, trainable: {n_trainable:,} ({100 * n_trainable / n_total:.1f}%)")
    exp.log_parameters({"trainable_params": n_trainable, "total_params": n_total})

    optimizer = torch.optim.AdamW(trainable, lr=cfg.peak_lr, weight_decay=cfg.weight_decay)
    start_lr = cfg.start_lr if cfg.start_lr is not None else cfg.peak_lr / cfg.warmup_steps
    if cfg.cosine_total_steps is not None:
        scheduler = warmup_cosine_scheduler(
            optimizer, cfg.warmup_steps, cfg.cosine_total_steps, cfg.peak_lr,
            start_lr=cfg.start_lr,
        )
        log.info(
            f"Optimizer: AdamW, lr={start_lr:.2e}\u2192{cfg.peak_lr:.2e}\u21920 "
            f"(cosine, {cfg.cosine_total_steps} steps), wd={cfg.weight_decay:.2e}"
        )
    else:
        scheduler = warmup_constant_scheduler(
            optimizer, cfg.warmup_steps, cfg.peak_lr,
            start_lr=cfg.start_lr,
        )
        log.info(f"Optimizer: AdamW, lr={start_lr:.2e}→{cfg.peak_lr:.2e} (constant), wd={cfg.weight_decay:.2e}")

    amp_ctx = (
        torch.autocast(device_type=cfg.device.type, dtype=torch.bfloat16)
        if cfg.amp else nullcontext()
    )
    log.info(f"AMP: {'bfloat16' if cfg.amp else 'disabled'}")
    log.info(f"Non-blocking transfers: {'enabled' if cfg.non_blocking_transfer else 'DISABLED (sync)'}")

    # === RESTORE MODEL WEIGHTS ===
    # Two sources: .pt checkpoint (ckpt_data) or HF Hub seed (hf_seed_state_dict)
    weights_to_load: dict[str, Tensor] | None = None
    if ckpt_data is not None:
        weights_to_load = ckpt_data["state_dict"]
    elif hf_seed_state_dict is not None:
        weights_to_load = hf_seed_state_dict

    if weights_to_load is not None:
        if is_rgb:
            # rgb_recon has no standardizers; require an exact match.
            model.load_state_dict(weights_to_load)
        else:
            assert isinstance(model, CanViTForPretraining)
            load_state_dict_flexible(model, weights_to_load)

    # === RESTORE OPTIMIZER/SCHEDULER (RESUME mode only) ===
    if ckpt_data is not None and not is_seeding:
        opt_state = ckpt_data["optimizer_state"]
        sched_state = ckpt_data["scheduler_state"]
        assert opt_state is not None, "Checkpoint missing optimizer_state — cannot resume"
        assert sched_state is not None, "Checkpoint missing scheduler_state — cannot resume"
        optimizer.load_state_dict(opt_state)
        scheduler.load_state_dict(sched_state)
        log.info(f"RESUME mode: restored optimizer+scheduler (step={sched_state['last_epoch']})")
    elif is_seeding:
        log.info("SEED mode: fresh optimizer+scheduler (step=0)")

    # Build training config history (tracks config across resumes)
    training_config_history: dict[str, dict] = {}
    if ckpt_data is not None:
        training_config_history = ckpt_data["training_config_history"] or {}
    now = datetime.now(UTC).isoformat()
    training_config_history[now] = flatten_dict(asdict(cfg))

    # Build provenance history (tracks git/host/slurm across resumes)
    provenance_history: dict[str, dict] = {}
    if ckpt_data is not None:
        provenance_history = ckpt_data.get("provenance_history") or {}
    provenance_history[now] = current_provenance()

    def make_ckpt_path(step: int) -> Path:
        """Generate versioned checkpoint path: {run_dir}/step-{step}.pt"""
        return run_dir / f"step-{step}.pt"

    def compute_raw_targets(images: Tensor, sz: int) -> NormFeatures:
        assert teacher is not None
        with amp_ctx:
            if images.shape[-1] != sz:
                images = torch.nn.functional.interpolate(
                    images, size=(sz, sz), mode="bilinear", align_corners=False
                )
            feats = teacher.forward_norm_features(images)
            return NormFeatures(patches=feats.patches.float(), cls=feats.cls.float())

    # Teacher feature standardizers exist only for distillation. rgb_recon reconstructs
    # raw pixels and has no standardizers.
    cls_norm: CLSStandardizer | None = None
    scene_norm: PatchStandardizer | None = None
    if not is_rgb:
        # Use model's own standardizers — their state is part of model.state_dict(),
        # so it travels correctly with HF Hub upload/download and checkpoint save/load.
        assert isinstance(model, CanViTForPretraining)
        # create_missing: a new canvas grid (e.g. higher-res continual pretraining)
        # gets fresh standardizers here; need_init below fills their statistics.
        cls_norm, scene_norm = model.standardizers(G, create_missing=True)

        need_init = cfg.reset_normalizer or not scene_norm.initialized
        if cfg.reset_normalizer:
            log.info("Reset normalizer: will re-init from shard")
        elif scene_norm.initialized:
            log.info("Standardizer stats loaded from model state_dict")

        if need_init:
            assert cfg.feature_base_dir is not None, "feature_base_dir required for standardizer init"
            shards_dir = cfg.feature_base_dir / cfg.teacher_name / str(cfg.scene_resolution) / "shards"
            shard_files = sorted(shards_dir.glob("*.pt"))
            assert shard_files, f"No shards in {shards_dir}"
            init_normalizer_stats_from_shard(
                shard_files[0], scene_norm, cls_norm, cfg.device, cfg.normalizer_max_samples,
            )

    log.info(
        f"Training: {cfg.n_full_start_branches} full + {cfg.n_random_start_branches} random branches,"
        f" chunk_size={cfg.chunk_size}, continue_prob={cfg.continue_prob}"
    )

    # EMA tracking for all metrics
    ema = EMATracker(alpha=cfg.ema_alpha)

    nb = cfg.non_blocking_transfer  # Ablation flag for async transfers

    def do_save(ckpt_path: Path, save_step: int) -> None:
        """Save a checkpoint (objective-aware) and repoint latest.pt."""
        ema_loss = ema.get("total_loss")
        save_checkpoint(
            ckpt_path, model, cfg.backbone_name,
            objective=cfg.objective,
            teacher_repo_id=None if is_rgb else cfg.teacher_repo_id,
            teacher_name=None if is_rgb else cfg.teacher_name,
            dataset=cfg.dataset,
            glimpse_grid_size=cfg.glimpse_grid_size,
            scene_resolution=cfg.scene_resolution,
            canvas_patch_grid_sizes=[G] if is_rgb else None,
            step=save_step, train_loss=ema_loss.item() if ema_loss is not None else None,
            comet_id=exp.get_key(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            training_config_history=training_config_history,
            provenance_history=provenance_history,
        )
        update_symlink(run_dir / "latest.pt", ckpt_path)

    def load_train_batch() -> TrainBatch:
        """Load distillation training batch (precomputed teacher features + standardization)."""
        assert scene_norm is not None and cls_norm is not None
        # non_blocking=True: CPU returns immediately, GPU ops serialize on same stream
        # Safe because we don't mutate source tensors after transfer
        images, raw_patches, raw_cls, labels = train_loader.next()
        images = images.to(cfg.device, non_blocking=nb)
        labels = labels.to(cfg.device, non_blocking=nb)
        # .float() for consistency - stored features may be fp16
        raw_patches = raw_patches.to(device=cfg.device, dtype=torch.float32, non_blocking=nb)
        raw_cls = raw_cls.to(device=cfg.device, dtype=torch.float32, non_blocking=nb)
        norm_patches = scene_norm(raw_patches)
        norm_cls = cls_norm(raw_cls.unsqueeze(1)).squeeze(1)
        return TrainBatch(images, labels, norm_patches, norm_cls, raw_patches, raw_cls)

    def load_rgb_images() -> Tensor:
        """rgb_recon needs only the scene images; teacher features in the shards are ignored."""
        images = train_loader.next()[0]
        return images.to(cfg.device, non_blocking=nb)

    # Timing accumulators for data vs GPU bottleneck analysis
    t_data_total = 0.0
    t_gpu_total = 0.0

    # Step semantics: step S = model state after S gradient updates
    # step=0: before any gradient (initial model)
    # Scheduler last_epoch tracks gradient updates done so far
    start_step = scheduler.last_epoch
    end_step = start_step + cfg.steps_per_job
    if ckpt_data is not None and not is_seeding and start_step == 0:
        log.error("!!! CHECKPOINT LOADED BUT start_step=0 - optimizer/scheduler state was not restored !!!")
    log.info(f"Starting training loop: steps {start_step} → {end_step}")
    model.train()  # Explicit: validate() restores, but be clear about initial state
    # NOTE: tqdm shows job-local progress (e.g. "199/5001"), but `step` is global
    pbar = tqdm(range(start_step, end_step + 1), desc="Training", unit="step")

    for step in pbar:
        batch: TrainBatch | None = None
        # Determine viz/curve based on validation count
        val_count = step // cfg.val_every
        do_pca = step % cfg.val_every == 0 and val_count % cfg.viz_every_n_vals == 0
        do_curves = step % cfg.val_every == 0 and val_count % cfg.curve_every_n_vals == 0

        # === VALIDATION/VIZ PHASE (state after `step` gradient updates) ===
        if step % cfg.val_every == 0:

            # Validation on val batch (always at val_every)
            val_images, val_labels = val_loader.next_batch_with_labels()
            val_images = val_images.to(cfg.device, non_blocking=nb)
            val_labels = val_labels.to(cfg.device, non_blocking=nb) if probe is not None else None
            try:
                with amp_ctx:
                    if is_rgb:
                        assert isinstance(model, CanViTForRGBReconstruction)
                        rgb_validate(
                            exp=exp, step=step, model=model, images=val_images,
                            canvas_grid_size=G, glimpse_size_px=glimpse_size_px,
                            n_steps=cfg.n_eval_viewpoints, min_viewpoint_scale=cfg.min_viewpoint_scale,
                        )
                    else:
                        assert isinstance(model, CanViTForPretraining)
                        assert scene_norm is not None and cls_norm is not None
                        validate(
                            exp=exp,
                            step=step,
                            model=model,
                            compute_raw_targets=compute_raw_targets,
                            scene_normalizer=scene_norm,
                            cls_normalizer=cls_norm,
                            images=val_images,
                            canvas_grid_size=G,
                            scene_size_px=scene_size,
                            glimpse_size_px=glimpse_size_px,
                            n_eval_viewpoints=cfg.n_eval_viewpoints,
                            min_viewpoint_scale=cfg.min_viewpoint_scale,
                            prefix="val",
                            probe=probe,
                            labels=val_labels,
                            log_curves=do_curves,
                            log_pca=do_pca,
                            teacher=teacher,
                            log_spatial_stats=cfg.log_spatial_stats,
                            teacher_name=cfg.teacher_name,
                        )
            except Exception:
                log.error(f"!!! VALIDATION FAILED at step {step} !!!\n{traceback.format_exc()}")

        # === SIGUSR1 CHECKPOINT ===
        global _checkpoint_requested
        if _checkpoint_requested:
            log.info(f"Saving signal-triggered checkpoint at step {step}")
            _checkpoint_requested = False
            do_save(make_ckpt_path(step), step)

        # === TRAINING PHASE (only for step < end_step) ===
        if step < end_step:
            t_data_start = time.perf_counter()
            loss_fn: LossFn
            viz_predict_fn = None
            viz_target = None
            if is_rgb:
                assert isinstance(model, CanViTForRGBReconstruction)
                images = load_rgb_images()
                pixel_target = patchify(images, model.patch_px)
                loss_fn = rgb_loss_fn(model=model, pixel_target=pixel_target)
                branch_metrics_fn = rgb_branch_metrics_fn(model=model, pixel_target=pixel_target)
                viz_predict_fn = model.predict_rgb_patches
                viz_target = pixel_target
            else:
                assert isinstance(model, CanViTForPretraining)
                assert scene_norm is not None and cls_norm is not None
                batch = load_train_batch()
                images = batch.images
                loss_fn = distillation_loss_fn(
                    model=model, scene_target=batch.scene_target, cls_target=batch.cls_target,
                    enable_scene_patches_loss=cfg.enable_scene_patches_loss,
                    enable_scene_cls_loss=cfg.enable_scene_cls_loss,
                )
                branch_metrics_fn = distillation_branch_metrics_fn(
                    model=model, scene_target=batch.scene_target, cls_target=batch.cls_target,
                    raw_scene_target=batch.raw_scene_target, raw_cls_target=batch.raw_cls_target,
                    scene_denorm=scene_norm.destandardize, cls_denorm=cls_norm.destandardize,
                )
                viz_predict_fn = model.predict_teacher_scene
                viz_target = batch.scene_target
            t_data = time.perf_counter() - t_data_start
            t_data_total += t_data

            optimizer.zero_grad()
            t_gpu_start = time.perf_counter()

            step_metrics = training_step(
                model=model,
                images=images,
                loss_fn=loss_fn,
                branch_metrics_fn=branch_metrics_fn,
                glimpse_size_px=glimpse_size_px,
                canvas_grid_size=G,
                n_full_start_branches=cfg.n_full_start_branches,
                n_random_start_branches=cfg.n_random_start_branches,
                chunk_size=cfg.chunk_size,
                continue_prob=cfg.continue_prob,
                min_viewpoint_scale=cfg.min_viewpoint_scale,
                amp_ctx=amp_ctx,
                collect_viz=do_pca,
                viz_predict_fn=viz_predict_fn,
                viz_target=viz_target,
            )

            grad_norm_t = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            t_gpu = time.perf_counter() - t_gpu_start
            t_gpu_total += t_gpu

            if step == start_step:
                log.info(f"First training_step took {t_gpu:.1f}s (includes compile)")

            # Update EMA for all metrics
            ema.update("total_loss", step_metrics.total_loss)
            ema.update("n_glimpses", torch.tensor(step_metrics.n_glimpses, dtype=torch.float32))
            for prefix, m in [("full", step_metrics.full_start), ("random", step_metrics.random_start)]:
                if m is None:
                    continue
                ema.update(f"{prefix}/loss", m.loss)
                for key, val in m.metrics.items():
                    ema.update(f"{prefix}/{key}", val)

            if step % cfg.log_every == 0:
                grad_norm = grad_norm_t.item()
                lr_val = scheduler.get_last_lr()[0]
                assert isinstance(lr_val, float)
                lr = lr_val

                # Log all EMA metrics
                metrics = {f"train/{k}": v.item() for k, v in ema.items()}
                metrics["train/lr"] = lr
                metrics["train/grad_norm"] = grad_norm
                metrics["train/continue_prob"] = cfg.continue_prob
                # Data vs GPU bottleneck: cumulative percentages
                data_pct = 0.0
                t_total_so_far = t_data_total + t_gpu_total
                if t_total_so_far > 0:
                    data_pct = t_data_total / t_total_so_far * 100
                    gpu_pct = t_gpu_total / t_total_so_far * 100
                    metrics["train/data_pct"] = data_pct
                    metrics["train/gpu_pct"] = gpu_pct
                exp.log_metrics(metrics, step=step)

                ema_loss = ema.get("total_loss")
                assert ema_loss is not None
                data_str = f"d={data_pct:.0f}%" if t_total_so_far > 0 else ""
                pbar.set_postfix_str(f"loss={ema_loss.item():.2e} grad={grad_norm:.2e} lr={lr:.2e} {data_str}")

            # Per-module grad norms (at val intervals, after training)
            if step % cfg.val_every == 0:
                for name, norm in grad_norms_by_module(model, depth=1).items():
                    exp.log_metric(f"grad_norm/{name}", norm, step=step)

            # Optuna pruning (skip step 0 - EMA not meaningful yet)
            if step > 0 and step % cfg.val_every == 0:
                ema_loss = ema.get("total_loss")
                assert ema_loss is not None
                trial.report(ema_loss.item(), step)
                if trial.should_prune():
                    exp.end()
                    raise optuna.TrialPruned()

            # Training-batch trajectory viz (same data as training, no recomputation).
            # Distillation: PCA of teacher-feature predictions. RGB: pixel reconstructions.
            if step_metrics.viz_data is not None:
                vd = step_metrics.viz_data
                assert vd.image.ndim == 3 and vd.image.shape[2] == 3, f"Expected [H,W,3], got {vd.image.shape}"
                H, W = vd.image.shape[:2]
                boxes = [vp.to_pixel_box(0, H, W) for vp in vd.viewpoints]
                names = [vp.name for vp in vd.viewpoints]
                scenes = [vs.predicted_scene for vs in vd.viz_samples]
                glimpses = [vs.glimpse for vs in vd.viz_samples]
                assert vd.initial_scene is not None
                if is_rgb:
                    fig = plot_multistep_recon(
                        full_img=vd.image,
                        target_patches=vd.target_features,
                        scenes=scenes,
                        glimpses=glimpses,
                        boxes=boxes,
                        names=names,
                        scene_grid_size=G,
                        initial_scene=vd.initial_scene,
                    )
                    log_figure(exp, fig, "train/reconstruction", step)
                else:
                    canvas_spatials = [vs.canvas_spatial for vs in vd.viz_samples]
                    fig = plot_multistep_pca(
                        full_img=vd.image,
                        teacher=vd.target_features,
                        scenes=scenes,
                        glimpses=glimpses,
                        boxes=boxes,
                        names=names,
                        scene_grid_size=G,
                        glimpse_grid_size=cfg.glimpse_grid_size,
                        initial_scene=vd.initial_scene,
                        hidden_spatials=canvas_spatials if canvas_spatials[0] is not None else None,
                        initial_hidden_spatial=vd.initial_canvas_spatial,
                    )
                    log_figure(exp, fig, "train/pca", step)

    # End-of-job checkpoint (always saved)
    do_save(make_ckpt_path(end_step), end_step)

    ema_loss = ema.get("total_loss")
    final_loss = ema_loss.item() if ema_loss is not None else float("inf")
    log.info(f"Final: train_ema={final_loss:.4f}")
    exp.end()
    return final_loss
