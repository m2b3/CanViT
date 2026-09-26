"""ImageNet-1k fine-tuning of CanViT-B on a Cloud TPU v6e slice with PyTorch/XLA SPMD (paper, Appendix D.4).

LP-FT: the flagship pretrained CanViT, its CLS readout fused with DINOv3 ViT-B/16's linear ImageNet-1k probe
(CanViTForImageClassification.from_pretrained_with_probe), then every parameter trained. Each scene is seen
through F-IID glimpses, with a cross-entropy loss after every glimpse and backpropagation through time truncated
to chunks of glimpses.
"""
# pyright: reportMissingImports=false
# (torch_xla is installed only in the tpu/ environment.)

import dataclasses
import logging
import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.backends as xla_backends
import torch_xla.debug.metrics as xla_metrics
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import tyro
from torch import Tensor
from torch.utils.data import DataLoader
from torch_xla.distributed.spmd import Mesh

from canvit_pytorch import CanViTForImageClassification, RecurrentState, Viewpoint
from canvit_pytorch.benchmarks.imagenet import NUM_TRAIN_IMAGES, NUM_VALIDATION_IMAGES
from canvit_pytorch.hub.repos import DINOV3_VITB16_IN1K_PROBE, FLAGSHIP, RELEASED_GLIMPSE_SIZE_PX
from canvit_pytorch.policies import POLICIES, PolicyName
from canvit_pytorch.specialize.in1k_tpu.checkpoint import (
    LATEST,
    Progress,
    load_checkpoint,
    load_classifier_weights,
    resume,
    save_checkpoint,
)
from canvit_pytorch.specialize.in1k_tpu.data import GlimpseBatch, GlimpseCollate, ShardSplit, glimpse_loader
from canvit_pytorch.specialize.in1k_tpu.figures import log_validation_predictions

log = logging.getLogger(__name__)

CANVAS_GRID_SIZE = 32
TRAINING_POLICY: PolicyName = "full_then_random"
VALIDATION_POLICY: PolicyName = "full_then_random"
"""Selects the best checkpoint and early stopping; eval_c2f adds a C2F validation."""
NUM_LOGGED_SAMPLES = 8

# Logged as environment diagnostics, except names containing a SECRET_SUBSTRINGS entry.
DIAGNOSTIC_ENV_PREFIXES = (
    "XLA_", "LIBTPU", "PJRT_", "PT_XLA", "TPU_", "OMP_", "MKL_", "TF_CPP", "SKYPILOT_", "PYTORCH_", "NEURON_", "GRPC_",
)
SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


@dataclass(frozen=True)
class FineTuningConfig:
    """Fine-tune CanViT-B for ImageNet-1k classification on a TPU slice. The defaults are the paper's run."""

    data_dir: Path
    """ImageNet-1k TFRecord shards, named train-* and validation-*."""
    batch_size: int = 256
    num_workers: int = 32
    peak_lr: float = 2.5e-5
    """The learning rate after linear warmup; it then decays to zero along a cosine."""
    weight_decay: float = 1e-4
    max_grad_norm: float | None = 1.0
    """Clip the gradient to this norm; None: no clipping."""
    label_smoothing: float = 0.1
    warmup_steps: int = 25_000
    epochs: int = 20
    num_glimpses: int = 4
    chunk_size: int = 4
    """Glimpses per chunk of truncated backpropagation through time; num_glimpses or more: full BPTT."""
    glimpse_size_px: int = RELEASED_GLIMPSE_SIZE_PX
    eval_num_glimpses: int | None = None
    """Glimpses per validation episode; None: num_glimpses."""
    eval_c2f: bool = False
    """Also validate on C2F episodes."""
    early_stop_delta: float | None = None
    """Stop when validation accuracy falls more than this below the best; None: never."""
    val_batches: int | None = None
    """Batches per validation; None: ceil(NUM_VALIDATION_IMAGES / batch_size)."""
    pre_training_val: bool = False
    """Validate before the first training step of a run that starts at step 0."""
    checkpoint_every: int = 200
    """Steps between saves of latest.pt, which also happen at every validation; 0: only at validations."""
    checkpoint_dir: Path | None = None
    """Where latest.pt, best.pt and init.pt go; a run whose checkpoint_dir holds latest.pt resumes from it."""
    init_from: Path | None = None
    """Start from this checkpoint's classifier weights at step 0 with a new optimizer, instead of resuming."""
    run_name: str | None = None
    log_every: int = 100


class ValidationLoader(NamedTuple):
    loader: DataLoader[Any]
    num_glimpses: int
    policy: PolicyName
    tag: str  # metric-name infix; "" for VALIDATION_POLICY


class StepStatistics(NamedTuple):
    loss: Tensor  # mean over glimpses
    num_correct: Tensor  # after the last glimpse
    grad_norm: Tensor  # before clipping
    num_correct_first: Tensor  # after the first glimpse
    loss_first: Tensor
    loss_last: Tensor


class CompileCounter:
    """Warns when XLA compiled a graph since the previous check: steady-state training compiles none."""

    def __init__(self) -> None:
        self.total = 0

    def check(self, step: int) -> None:
        data = xla_metrics.metric_data("CompileTime")
        if data is None:
            return
        new = data[0] - self.total
        self.total = data[0]
        if new:
            log.warning("xla_recompile step=%d compiles_since_last_log=%d total_compiles=%d", step, new, self.total)


_SHARDINGS_LOGGED: set[str] = set()


def log_sharding_once(tensor: Tensor, name: str) -> None:
    if name in _SHARDINGS_LOGGED:
        return
    _SHARDINGS_LOGGED.add(name)
    try:
        spec = torch_xla._XLAC._get_xla_sharding_spec(tensor)
    except Exception as error:  # noqa: BLE001  diagnostics only
        spec = f"<error: {error}>"
    log.info("sharding[%s] shape=%s dtype=%s spec=%s", name, tuple(tensor.shape), tensor.dtype, spec)


def log_environment() -> None:
    """Versions, precision, allowlisted environment variables with secrets redacted, device count."""
    import torchvision

    log.info("─── environment diagnostics ───")
    log.info("torch        = %s  (%s)", torch.__version__, torch.__file__)
    log.info("torchvision  = %s", torchvision.__version__)
    log.info("torch_xla    = %s  (%s)", torch_xla.__version__, torch_xla.__file__)
    try:
        import libtpu

        log.info("libtpu       = %s", getattr(libtpu, "__version__", "?"))
    except Exception as error:  # noqa: BLE001  diagnostics only
        log.info("libtpu       = (import failed: %s)", error)
    log.info("mat_mul_prec = %s (torch_xla)", xla_backends.get_mat_mul_precision())
    log.info("f32 matmul   = %s (torch core)", torch.get_float32_matmul_precision())
    log.info("CPU count    = %s", os.cpu_count())
    log.info("device count = %d (xr.global_runtime_device_count)", xr.global_runtime_device_count())
    log.info("─── env vars (allowlisted) ───")
    for name in sorted(os.environ):
        if name.startswith(DIAGNOSTIC_ENV_PREFIXES):
            redacted = any(secret in name.upper() for secret in SECRET_SUBSTRINGS)
            log.info("  %s = %s", name, "<redacted>" if redacted else os.environ[name])
    log.info("─── end diagnostics ───")


def load_classifier(device: torch.device) -> CanViTForImageClassification:
    start = time.perf_counter()
    classifier = CanViTForImageClassification.from_pretrained_with_probe(
        pretrained_repo=FLAGSHIP, probe_repo=DINOV3_VITB16_IN1K_PROBE,
    ).to(device)
    log.info("Loaded classifier: %s params, %.1fs",
             f"{sum(p.numel() for p in classifier.parameters()):,}", time.perf_counter() - start)
    return classifier


def initial_state(classifier: CanViTForImageClassification, batch_size: int, mesh: Mesh | None) -> RecurrentState:
    state = classifier.init_state(batch_size=batch_size, canvas_grid_size=CANVAS_GRID_SIZE)
    if mesh is not None:
        xs.mark_sharding(state.canvas, mesh, ("data", None, None))
        xs.mark_sharding(state.recurrent_cls, mesh, ("data", None, None))
    log_sharding_once(state.canvas, "state.canvas")
    log_sharding_once(state.recurrent_cls, "state.recurrent_cls")
    return state


def to_device(batch: GlimpseBatch, device: torch.device, mesh: Mesh | None) -> GlimpseBatch:
    """The batch on the device; glimpses and labels sharded over the batch dimension."""
    batch = GlimpseBatch(*(tensor.to(device) for tensor in batch))
    if mesh is not None:
        xs.mark_sharding(batch.glimpses, mesh, (None, "data", None, None, None))
        xs.mark_sharding(batch.labels, mesh, ("data",))
    log_sharding_once(batch.glimpses, "batch.glimpses")
    log_sharding_once(batch.labels, "batch.labels")
    return batch


def start_comet(*, run_name: str | None, previous_key: str | None) -> Any:
    """A Comet experiment, continued when previous_key is set; None without COMET_API_KEY."""
    if not os.environ.get("COMET_API_KEY"):
        return None
    import comet_ml

    experiment_config = comet_ml.ExperimentConfig(auto_metric_logging=False, name=run_name)
    if previous_key is None:
        return comet_ml.start(experiment_config=experiment_config)
    log.info("Continuing Comet experiment: %s", previous_key)
    return comet_ml.start(experiment_key=previous_key, experiment_config=experiment_config)


def lr_multiplier(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup from 0, then cosine decay to 0: the LambdaLR factor of peak_lr."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_step(
    *, classifier: CanViTForImageClassification, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler, device: torch.device, batches: Iterator[GlimpseBatch],
    config: FineTuningConfig, mesh: Mesh | None,
) -> StepStatistics:
    """One optimizer step on one batch: a loss after every glimpse, divided by the number of glimpses,
    backpropagated chunk by chunk (truncated BPTT)."""
    glimpses, labels, centers, scales = to_device(next(batches), device, mesh)
    num_glimpses, batch_size = glimpses.shape[:2]
    state = initial_state(classifier, batch_size, mesh)

    chunk_loss = torch.zeros((), device=device)
    total_loss = torch.zeros((), device=device)
    num_correct_first = torch.zeros((), dtype=torch.long, device=device)
    loss_first = torch.zeros((), device=device)
    for g in range(num_glimpses):
        logits, state = classifier(glimpse=glimpses[g], state=state, viewpoint=Viewpoint(centers=centers[g], scales=scales[g]))
        glimpse_loss = F.cross_entropy(logits, labels, label_smoothing=config.label_smoothing)
        chunk_loss = chunk_loss + glimpse_loss
        total_loss = total_loss + glimpse_loss.detach()
        if g == 0:
            num_correct_first = (logits.argmax(dim=-1) == labels).sum()
            loss_first = glimpse_loss.detach()
        is_last = g == num_glimpses - 1
        if (g + 1) % config.chunk_size == 0 or is_last:
            (chunk_loss / num_glimpses).backward()
            if not is_last:
                # Dispatch each chunk's graph now, so memory grows with chunk_size, not num_glimpses. With full
                # BPTT the sync at the end of the step runs forward, backward and optimizer as one graph.
                torch_xla.sync()
                state = state.detach()
                chunk_loss = torch.zeros((), device=device)

    loss = total_loss / num_glimpses
    num_correct = (logits.argmax(dim=-1) == labels).sum()  # pyright: ignore[reportPossiblyUnboundVariable]  (num_glimpses >= 1)
    loss_last = glimpse_loss.detach()  # pyright: ignore[reportPossiblyUnboundVariable]  (num_glimpses >= 1)
    grad_norm = torch.cat([p.grad.detach().float().flatten() for p in classifier.parameters() if p.grad is not None]).norm(2)
    if config.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    torch_xla.sync()
    return StepStatistics(
        loss=loss, num_correct=num_correct, grad_norm=grad_norm, num_correct_first=num_correct_first,
        loss_first=loss_first, loss_last=loss_last,
    )


@torch.no_grad()
def validate(
    *, classifier: CanViTForImageClassification, device: torch.device, validation: ValidationLoader,
    num_batches: int, mesh: Mesh | None, experiment: Any, step: int, log_samples: bool,
) -> dict[str, float]:
    """Accuracy and mean loss after each glimpse over num_batches batches, and the wall time."""
    classifier.eval()
    batches = iter(validation.loader)
    num_scenes = 0
    start = time.perf_counter()
    num_correct: list[Tensor] = []
    loss_sums: list[Tensor] = []
    sample_glimpses = sample_logits = sample_labels = None
    for i in range(num_batches):
        batch = next(batches)
        if i == 0 and log_samples:
            sample_glimpses = batch.glimpses[0, :NUM_LOGGED_SAMPLES].clone()
        glimpses, labels, centers, scales = to_device(batch, device, mesh)
        num_glimpses, batch_size = glimpses.shape[:2]
        if not num_correct:
            num_correct = [torch.zeros((), dtype=torch.long, device=device) for _ in range(num_glimpses)]
            loss_sums = [torch.zeros((), device=device) for _ in range(num_glimpses)]
        state = initial_state(classifier, batch_size, mesh)
        for g in range(num_glimpses):
            logits, state = classifier(glimpse=glimpses[g], state=state, viewpoint=Viewpoint(centers=centers[g], scales=scales[g]))
            num_correct[g] += (logits.argmax(dim=-1) == labels).sum()
            loss_sums[g] += F.cross_entropy(logits, labels)
            if i == 0 and g == 0 and log_samples:
                sample_logits = logits[:NUM_LOGGED_SAMPLES].float().cpu()
                sample_labels = labels[:NUM_LOGGED_SAMPLES].cpu()
        num_scenes += batch_size
    torch_xla.sync(wait=True)
    elapsed = time.perf_counter() - start
    classifier.train()

    infix = f"_{validation.tag}" if validation.tag else ""
    metrics = {f"val/time{infix}_sec": elapsed}
    for g in range(len(num_correct)):
        metrics[f"val/accuracy{infix}_t{g}"] = num_correct[g].item() / num_scenes
        metrics[f"val/loss{infix}_t{g}"] = loss_sums[g].item() / num_batches
    if experiment and sample_glimpses is not None and sample_logits is not None and sample_labels is not None:
        log_validation_predictions(
            experiment=experiment, step=step, glimpses=sample_glimpses, logits=sample_logits, labels=sample_labels,
        )
    return metrics


def validate_all(
    *, classifier: CanViTForImageClassification, device: torch.device, validations: list[ValidationLoader],
    num_batches: int, mesh: Mesh | None, experiment: Any, step: int, label: str, log_samples: bool,
) -> float:
    """Run every validation and log it; return the accuracy after the last glimpse of the VALIDATION_POLICY one."""
    primary_accuracy = -1.0
    for validation in validations:
        is_primary = validation.tag == ""
        metrics = validate(
            classifier=classifier, device=device, validation=validation, num_batches=num_batches, mesh=mesh,
            experiment=experiment, step=step, log_samples=log_samples and is_primary,
        )
        infix = f"_{validation.tag}" if validation.tag else ""
        accuracy = metrics[f"val/accuracy{infix}_t{validation.num_glimpses - 1}"]
        curve = " → ".join(f"{metrics[f'val/accuracy{infix}_t{g}']:.1%}" for g in range(validation.num_glimpses))
        log.info("  VAL %s (%s, N=%d) | acc %.1f%% | curve: [%s] | %.1fs", label,
                 POLICIES[validation.policy].paper_name, validation.num_glimpses, accuracy * 100, curve,
                 metrics[f"val/time{infix}_sec"])
        if experiment:
            experiment.log_metrics(metrics, step=step)
        if is_primary:
            primary_accuracy = accuracy
    return primary_accuracy


def train(config: FineTuningConfig) -> float:
    """Fine-tune; return the best validation accuracy after the last glimpse."""
    train_start = time.perf_counter()
    log_environment()

    num_devices = xr.global_runtime_device_count()
    mesh = None
    if num_devices > 1:
        xr.use_spmd()
        mesh = Mesh(np.arange(num_devices), (num_devices,), ("data",))
        log.info("SPMD: %d devices, mesh=(%d,), partition='data'", num_devices, num_devices)
    else:
        log.info("No SPMD: only %d device; mark_sharding skipped", num_devices)

    device = torch_xla.device()
    steps_per_epoch = NUM_TRAIN_IMAGES // config.batch_size
    total_steps = config.epochs * steps_per_epoch
    # XXX: each validation reads an endless stream in which every worker cycles through its own shards, so these
    # batches are not exactly one pass over the validation set: some scenes repeat and others are missed.
    num_val_batches = config.val_batches or math.ceil(NUM_VALIDATION_IMAGES / config.batch_size)
    eval_num_glimpses = config.eval_num_glimpses or config.num_glimpses
    log.info("Device: %s", device)
    log.info("Plan: %d epochs × %d steps/epoch = %d total, N=%d glimpses, chunk=%d",
             config.epochs, steps_per_epoch, total_steps, config.num_glimpses, config.chunk_size)

    start = time.perf_counter()

    def loader(split: ShardSplit, policy: PolicyName, num_glimpses: int) -> DataLoader[Any]:
        collate = GlimpseCollate(
            policy=policy, num_glimpses=num_glimpses, glimpse_size_px=config.glimpse_size_px,
            canvas_grid_size=CANVAS_GRID_SIZE,
        )
        return glimpse_loader(
            data_dir=config.data_dir, split=split, collate=collate, batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

    def validation(policy: PolicyName, tag: str) -> ValidationLoader:
        return ValidationLoader(
            loader=loader("validation", policy, eval_num_glimpses), num_glimpses=eval_num_glimpses, policy=policy, tag=tag,
        )

    training_batches = iter(loader("train", TRAINING_POLICY, config.num_glimpses))
    validations = [validation(VALIDATION_POLICY, "")] + ([validation("coarse_to_fine", "c2f")] if config.eval_c2f else [])
    log.info("DataLoaders ready (B=%d, %dw, eval_N=%d) [%.1fs]",
             config.batch_size, config.num_workers, eval_num_glimpses, time.perf_counter() - start)

    start = time.perf_counter()
    classifier = load_classifier(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=config.peak_lr, weight_decay=config.weight_decay)
    log.info("Model + optimizer ready [%.1fs]", time.perf_counter() - start)

    progress = Progress(step=0, best_val_acc=-1.0, comet_key=None)
    if config.init_from is not None:
        start = time.perf_counter()
        state = load_checkpoint(config.init_from)
        load_classifier_weights(state, classifier, device)
        log.info("Init from step %s [%.1fs]. Training from step 0.", state["step"], time.perf_counter() - start)
    elif config.checkpoint_dir is not None:
        start = time.perf_counter()
        progress = resume(checkpoint_dir=config.checkpoint_dir, classifier=classifier, optimizer=optimizer, device=device)
        log.info("Resume [%.1fs] → step %d, best_val_acc=%.4f", time.perf_counter() - start, progress.step, progress.best_val_acc)
    start_step, best_val_acc = progress.step, progress.best_val_acc

    start = time.perf_counter()
    experiment = start_comet(run_name=config.run_name, previous_key=progress.comet_key)
    if experiment:
        log.info("Comet [%.1fs]: %s (%s)", time.perf_counter() - start, experiment.get_key(), config.run_name or "auto")
        experiment.log_parameters(dataclasses.asdict(config) | {
            "total_steps": total_steps, "num_parameters": sum(p.numel() for p in classifier.parameters()),
            "start_step": start_step, "eval_num_glimpses": eval_num_glimpses,
            "sky_task_id": os.environ.get("SKYPILOT_TASK_ID", ""), "sky_user": os.environ.get("SKYPILOT_USER", ""),
        })
    else:
        log.info("Comet skipped — COMET_API_KEY not set")
    comet_key = experiment.get_key() if experiment else None

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, warmup_steps=config.warmup_steps, total_steps=total_steps),
    )
    for _ in range(start_step):
        scheduler.step()

    def step_once() -> StepStatistics:
        return train_step(
            classifier=classifier, optimizer=optimizer, scheduler=scheduler, device=device,
            batches=training_batches, config=config, mesh=mesh,
        )

    # XXX: this compiling step is a full training step on a real batch (an optimizer and a scheduler step), not
    # counted in `step`: a run takes total_steps + 1 steps, one more per resumption, and init.pt holds its result.
    log.info("Compiling XLA graph (N=%d, chunk=%d)...", config.num_glimpses, config.chunk_size)
    start = time.perf_counter()
    step_once()
    torch_xla.sync(wait=True)
    log.info("Compiled [%.1fs]", time.perf_counter() - start)

    def checkpoint(step: int, filename: str = LATEST) -> None:
        if config.checkpoint_dir is not None:
            save_checkpoint(
                checkpoint_dir=config.checkpoint_dir, filename=filename,
                progress=Progress(step=step, best_val_acc=best_val_acc, comet_key=comet_key),
                classifier=classifier, optimizer=optimizer, wait_for_device=lambda: torch_xla.sync(wait=True),
            )

    def validate_now(step: int, label: str) -> float:
        return validate_all(
            classifier=classifier, device=device, validations=validations, num_batches=num_val_batches, mesh=mesh,
            experiment=experiment, step=step, label=label, log_samples=True,
        )

    if start_step == 0:
        checkpoint(0)
        checkpoint(0, filename="init.pt")
    if config.pre_training_val and start_step == 0:
        log.info("Pre-training validation (step 0)...")
        validate_now(0, "step 0")

    log.info("Training from step %d to %d [startup %.1fs]", start_step, total_steps, time.perf_counter() - train_start)
    compile_counter = CompileCounter()
    window = StepStatistics(*(
        torch.zeros((), dtype=torch.long if field.startswith("num_correct") else torch.float32, device=device)
        for field in StepStatistics._fields
    ))  # sums since the last log
    window_start, window_steps = time.perf_counter(), 0
    step = start_step
    for step in range(start_step, total_steps):
        for window_sum, value in zip(window, step_once(), strict=True):
            window_sum += value.detach()
        window_steps += 1

        if step % config.log_every == 0 and step > start_step:
            torch_xla.sync(wait=True)
            compile_counter.check(step)
            elapsed = time.perf_counter() - window_start
            window_scenes = window_steps * config.batch_size
            last = config.num_glimpses - 1
            metrics = {
                "loss": (window.loss / window_steps).item(),
                "accuracy": window.num_correct.item() / window_scenes,
                "grad_norm": (window.grad_norm / window_steps).item(),
                "scenes_per_sec": window_scenes / elapsed,
                "glimpses_per_sec": window_scenes / elapsed * config.num_glimpses,
                "ms_per_step": elapsed / window_steps * 1000,
                "lr": scheduler.get_last_lr()[0],
                "accuracy_t0": window.num_correct_first.item() / window_scenes,
                "loss_t0": (window.loss_first / window_steps).item(),
                f"loss_t{last}": (window.loss_last / window_steps).item(),
            }
            log.info("step %6d ep %.2f | loss %.4f (t0=%.3f t%d=%.3f) | acc t0=%.1f%% t%d=%.1f%% | gnorm %.2e | "
                     "lr %.2e | %.0f sc/s", step, step / steps_per_epoch, metrics["loss"], metrics["loss_t0"], last,
                     metrics[f"loss_t{last}"], metrics["accuracy_t0"] * 100, last, metrics["accuracy"] * 100,
                     metrics["grad_norm"], metrics["lr"], metrics["scenes_per_sec"])
            if experiment:
                experiment.log_metrics(metrics, step=step)
            for window_sum in window:
                window_sum.zero_()
            window_start, window_steps = time.perf_counter(), 0

        if config.checkpoint_every > 0 and step > 0 and step % config.checkpoint_every == 0:
            checkpoint(step)

        if step > 0 and step % steps_per_epoch == 0:
            val_acc = validate_now(step, f"ep {step // steps_per_epoch}")
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                log.info("  New best: %.1f%%", best_val_acc * 100)
            checkpoint(step)
            if is_best:
                checkpoint(step, filename="best.pt")
            if config.early_stop_delta is not None and best_val_acc - val_acc > config.early_stop_delta:
                log.info("  EARLY STOP: val %.1f%% < best %.1f%% by %.1fpp (threshold %.1fpp)", val_acc * 100,
                         best_val_acc * 100, (best_val_acc - val_acc) * 100, config.early_stop_delta * 100)
                break

    torch_xla.sync(wait=True)
    final_step = step + 1
    val_acc = validate_now(final_step, "final")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        log.info("  New best: %.1f%%", best_val_acc * 100)
        checkpoint(final_step, filename="best.pt")
    checkpoint(final_step)

    if experiment:
        experiment.end()
    log.info("Done. best_val_acc=%.1f%%", best_val_acc * 100)
    return best_val_acc


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", level=logging.INFO, force=True)
    train(tyro.cli(FineTuningConfig, description=__doc__))


if __name__ == "__main__":
    main()
