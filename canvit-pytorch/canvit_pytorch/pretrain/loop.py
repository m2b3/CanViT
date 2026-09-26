"""A pretraining job: resume the run, train steps_per_job steps, save, exit.

A run spans many jobs (a SLURM array). Each job resumes from the run's latest
checkpoint and continues the same Comet experiment. A crash writes FAILED in
the run directory and cancels the rest of the array, so a broken run does not
crash in a loop; delete the marker after fixing the cause.
"""

import io
import json
import logging
import os
import signal
import subprocess
import time
import traceback
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import comet_ml
import torch
import torch._functorch.config

from canvit_pytorch.model.pretraining import CanViTForPretraining
from canvit_pytorch.policies import POLICIES
from canvit_pytorch.pretrain import checkpoint
from canvit_pytorch.pretrain.config import PretrainingConfig
from canvit_pytorch.pretrain.data import TrainingBatches, ValidationBatches
from canvit_pytorch.pretrain.features.shard import load_shard
from canvit_pytorch.pretrain.loss import DistillationTargets
from canvit_pytorch.pretrain.monitor.figures import figure_png, rollout_figure
from canvit_pytorch.pretrain.monitor.training import MetricEMA, grad_norms_by_module
from canvit_pytorch.pretrain.monitor.validation import (
    TEACHER_CLASSIFIER_SIZE_PX,
    load_teacher_classifier,
    validate,
)
from canvit_pytorch.pretrain.schedule import warmup_then_constant
from canvit_pytorch.pretrain.step import training_step
from canvit_pytorch.provenance import provenance
from canvit_pytorch.teacher import TEACHER_REPO, DINOv3Teacher, load_teacher

log = logging.getLogger(__name__)


def train(config: PretrainingConfig) -> None:
    run_dir = config.checkpoints_dir / config.run_name
    failed_marker = run_dir / "FAILED"
    if failed_marker.exists():
        cancel_slurm_array()
        raise RuntimeError(f"{failed_marker} exists: an earlier job of this run crashed. Fix the cause, then delete it.")
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        _train_job(config, run_dir)
    except Exception:
        failed_marker.write_text(f"{datetime.now(UTC).isoformat()}\n{traceback.format_exc()}")
        cancel_slurm_array()
        raise


def cancel_slurm_array() -> None:
    if (job := os.environ.get("SLURM_ARRAY_JOB_ID")) is not None:
        log.error(f"Cancelling SLURM array {job}")
        subprocess.run(["scancel", job], check=False)


def _configure_torch(device: torch.device) -> None:
    # Backward runs outside autocast. torch.compile otherwise assumes backward runs under the forward's
    # autocast state and computes wrong gradients.
    torch._functorch.config.backward_pass_autocast = "off"  # pyright: ignore[reportAttributeAccessIssue]
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        # Flash attention only: fail rather than fall back to a slower kernel.
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)


def _fit_standardizers(model: CanViTForPretraining, shards_dir: Path, device: torch.device) -> None:
    """Per-position statistics of the teacher's features over the first shard's scenes."""
    path = sorted(shards_dir.glob("*.pt"))[0]
    shard = load_shard(path)
    model.teacher_patch_standardizer.fit(shard["patches"].float().to(device))
    model.teacher_cls_standardizer.fit(shard["cls"].float().to(device).unsqueeze(1))
    log.info(f"Fitted the target standardizers on the {len(shard['paths'])} scenes of {path}")


def _start_experiment(resume_key: str | None) -> comet_ml.CometExperiment:
    experiment_config = comet_ml.ExperimentConfig(auto_metric_logging=False)
    if resume_key is None:
        return comet_ml.start(mode="create", experiment_config=experiment_config)
    return comet_ml.start(mode="get", experiment_key=resume_key, experiment_config=experiment_config)


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat |= _flatten(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = value if isinstance(value, int | float | bool | str | None) else str(value)
    return flat


def _train_job(config: PretrainingConfig, run_dir: Path) -> None:
    device = torch.device(config.device)
    _configure_torch(device)
    teacher = load_teacher(TEACHER_REPO, device)
    scene_size_px = config.canvas_grid_size * teacher.patch_size
    if scene_size_px != TEACHER_CLASSIFIER_SIZE_PX:
        log.warning(f"Validation top-1 applies a {TEACHER_CLASSIFIER_SIZE_PX} px classifier to {scene_size_px} px scenes")

    model = CanViTForPretraining(
        canvit_config=config.model, teacher_dim=teacher.embed_dim, teacher_patch_grid=config.canvas_grid_size,
    ).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.peak_lr, weight_decay=config.weight_decay)
    scheduler = warmup_then_constant(
        optimizer, start_lr=config.start_lr, peak_lr=config.peak_lr, warmup_steps=config.warmup_steps,
    )

    job = {"config": json.loads(json.dumps(asdict(config), default=str)), "provenance": provenance(device)}
    if (latest := checkpoint.latest(run_dir)) is None:
        log.info(f"Starting run {run_dir}")
        _fit_standardizers(model, config.shards_dir, device)
        start_step, jobs, experiment_key = 0, [job], None
    else:
        log.info(f"Resuming from {latest}")
        saved = checkpoint.load(latest, device=device)
        assert saved["canvit_config"] == asdict(config.model), (
            f"the run trained {saved['canvit_config']}; this job's config says {asdict(config.model)}"
        )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_step, jobs, experiment_key = saved["step"], [*saved["jobs"], job], saved["comet_experiment_key"]
        assert scheduler.last_epoch == start_step, (scheduler.last_epoch, start_step)

    end_step = min(start_step + config.steps_per_job, config.total_steps)
    if start_step >= end_step:
        log.info(f"The run is complete at step {start_step}")
        return
    experiment = _start_experiment(experiment_key)
    log.info(f"Comet experiment: {experiment.url}")
    experiment.log_parameters(_flatten(asdict(config)) | {"job": len(jobs)})

    def save(step: int) -> None:
        checkpoint.save(checkpoint.TrainingCheckpoint(
            format=checkpoint.CHECKPOINT_FORMAT, model=model.state_dict(), **checkpoint.model_init_kwargs(model),
            teacher_repo=TEACHER_REPO, dataset=config.dataset, scene_size_px=scene_size_px,
            glimpse_size_px=config.glimpse_size_px, step=step,
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            comet_experiment_key=experiment.get_key(), jobs=jobs,
        ), run_dir)

    save_requested = False

    def request_save(signum: int, frame: object) -> None:
        nonlocal save_requested
        save_requested = True

    signal.signal(signal.SIGUSR1, request_save)  # SLURM sends it before a time limit or preemption

    if config.compile:
        model.canvit.compile()
        teacher.compile()
    autocast = torch.autocast(device.type, dtype=torch.bfloat16) if config.amp else nullcontext()
    training_batches = TrainingBatches(
        shards_dir=config.shards_dir, images_dir=config.images_dir, scene_size_px=scene_size_px,
        teacher_repo=TEACHER_REPO, batch_size=config.batch_size, num_workers=config.num_workers, start_step=start_step,
    )
    validation_batches = ValidationBatches(
        validation_dir=config.validation_dir, scene_size_px=scene_size_px,
        batch_size=config.batch_size, num_workers=config.num_workers,
    )
    teacher_classifier = load_teacher_classifier(device)
    averages = MetricEMA(config.metric_ema_alpha)
    seconds_loading = seconds_computing = 0.0

    log.info(f"Training steps {start_step} to {end_step} of {config.total_steps}")
    model.train()
    for step in range(start_step, end_step):
        # Step N sees the model after N updates.
        if step % config.validate_every == 0:
            _validate_and_log(
                config=config, experiment=experiment, step=step, model=model, teacher=teacher,
                teacher_classifier=teacher_classifier, batches=validation_batches, autocast=autocast,
            )
        if save_requested:
            save(step)
            save_requested = False

        started = time.perf_counter()
        images, teacher_patches, teacher_cls = training_batches.next()
        images = images.to(device, non_blocking=True)
        targets = DistillationTargets.standardize(
            model,
            patches=teacher_patches.to(device, dtype=torch.float32, non_blocking=True),
            cls=teacher_cls.to(device, dtype=torch.float32, non_blocking=True),
        )
        loaded = time.perf_counter()
        optimizer.zero_grad()
        metrics = training_step(
            model=model, images=images, targets=targets, rollout_policies=config.rollout_policies,
            tbptt_chunk_glimpses=config.tbptt_chunk_glimpses, stop_probability=config.stop_probability,
            glimpse_size_px=config.glimpse_size_px, canvas_grid_size=config.canvas_grid_size,
            enable_teacher_patch_loss=config.enable_teacher_patch_loss,
            enable_teacher_cls_loss=config.enable_teacher_cls_loss, autocast=autocast,
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        seconds_loading += loaded - started
        seconds_computing += time.perf_counter() - loaded

        averages.update({"loss": metrics.loss, "num_glimpses": torch.tensor(float(metrics.num_glimpses))} | {
            f"{POLICIES[policy].paper_name}/{name}": value
            for policy, values in metrics.by_policy.items() for name, value in values.items()
        })
        if step % config.log_every == 0:
            smoothed = averages.read()
            loading_share = seconds_loading / (seconds_loading + seconds_computing)
            experiment.log_metrics(
                {f"train/{name}": value for name, value in smoothed.items()}
                | {"train/lr": scheduler.get_last_lr()[0], "train/grad_norm": grad_norm.item(),
                   "train/loading_share": loading_share},
                step=step,
            )
            log.info(f"step {step}: loss {smoothed['loss']:.4f}, grad norm {grad_norm.item():.3f}, "
                     f"lr {scheduler.get_last_lr()[0]:.2e}, {loading_share:.0%} of time loading data")
        if step % config.validate_every == 0:
            experiment.log_metrics({f"grad_norm/{name}": v for name, v in grad_norms_by_module(model, depth=2).items()}, step=step)

    save(end_step)
    experiment.end()


def _validate_and_log(
    *, config: PretrainingConfig, experiment: comet_ml.CometExperiment, step: int, model: CanViTForPretraining,
    teacher: DINOv3Teacher, teacher_classifier: torch.nn.Linear, batches: ValidationBatches,
    autocast: AbstractContextManager,
) -> None:
    """Monitoring must not end a paid-for job: a failure here is logged with its traceback and training continues."""
    try:
        images, labels = batches.next()
        device = next(model.parameters()).device
        rollout = validate(
            model=model, teacher=teacher, teacher_classifier=teacher_classifier,
            images=images.to(device), labels=labels.to(device), policy_name=config.validation_policy,
            num_glimpses=config.validation_glimpses, glimpse_size_px=config.glimpse_size_px,
            canvas_grid_size=config.canvas_grid_size, autocast=autocast,
        )
        metrics = {"val/teacher_in1k_top1": rollout.teacher_in1k_top1}
        for t, values in enumerate(rollout.per_glimpse):
            metrics |= {f"val/{name}_t{t}": value for name, value in values.items()}
        metrics |= {f"val/{name}": value for name, value in rollout.per_glimpse[-1].items()}
        experiment.log_metrics(metrics, step=step)
        if step % config.figure_every == 0:
            inputs = rollout.figure_inputs
            figure = rollout_figure(
                image=inputs.image, viewpoints=inputs.viewpoints, glimpse_size_px=config.glimpse_size_px,
                teacher_patches=inputs.teacher_patches, predicted_patches=inputs.predicted_patches,
                canvas_patches=inputs.canvas_patches,
            )
            experiment.log_image(io.BytesIO(figure_png(figure)), name="val/rollout", step=step)
    except Exception:
        log.exception(f"Validation failed at step {step}")
