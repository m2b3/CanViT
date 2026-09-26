"""CanViT ImageNet-1k finetuning on TPU v6e (PyTorch/XLA SPMD).

Trains CanViTForImageClassification (CanViT + LN → Linear head) on IN1k.
Head initialized from fused DINOv3 probe. All parameters trainable.
"""

import logging
import time as _time

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                    level=logging.INFO, force=True)
log = logging.getLogger("train")

_PYTHON_START = _time.perf_counter()
log.info("Python interpreter started")

import argparse
import math
import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.backends as xla_backends
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from canvit_pytorch import CanViTForImageClassification, Viewpoint
from torch_xla.distributed.spmd import Mesh

from .shared import CANVAS_GRID, IMAGENET_MEAN, IMAGENET_STD, load_classifier, make_multi_glimpse_dataloader
from .training_utils import ValLoader, apply_model_weights, load_checkpoint, make_lr_lambda, maybe_resume, save_checkpoint, should_early_stop
from .viz import log_val_samples as _log_val_samples

TRAIN_IMAGES = 1_281_167
VAL_IMAGES = 50_000

log.info("imports done [%.1fs]", time.perf_counter() - _PYTHON_START)


_ENV_ALLOWLIST_PREFIXES = (
    "XLA_", "LIBTPU", "PJRT_", "PT_XLA", "TPU_", "OMP_", "MKL_", "TF_CPP",
    "SKYPILOT_", "PYTORCH_", "NEURON_", "GRPC_",
)
# Never log anything matching these substrings even if prefix matches — defense
# in depth so --secret values can't slip in via an aliased prefix.
_ENV_SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _log_environment_diagnostics() -> None:
    """Dump versions + precision + env vars + device topology. Secret-bearing env vars are filtered."""
    import torchvision
    log.info("─── environment diagnostics ───")
    log.info("torch        = %s  (%s)", torch.__version__, torch.__file__)
    log.info("torchvision  = %s", torchvision.__version__)
    log.info("torch_xla    = %s  (%s)", torch_xla.__version__, torch_xla.__file__)
    try:
        import libtpu
        log.info("libtpu       = %s", getattr(libtpu, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        log.info("libtpu       = (import failed: %s)", e)
    log.info("mat_mul_prec = %s (torch_xla)", xla_backends.get_mat_mul_precision())
    log.info("f32 matmul   = %s (torch core)", torch.get_float32_matmul_precision())
    log.info("CPU count    = %s", os.cpu_count())
    log.info("device count = %d (xr.global_runtime_device_count)", xr.global_runtime_device_count())
    log.info("─── env vars (allowlisted) ───")
    for k in sorted(os.environ):
        if not k.startswith(_ENV_ALLOWLIST_PREFIXES):
            continue
        if any(s in k.upper() for s in _ENV_SECRET_SUBSTRINGS):
            log.info("  %s = <redacted>", k)
            continue
        log.info("  %s = %s", k, os.environ[k])
    log.info("─── end diagnostics ───")


def _log_sharding_spec(tensor: torch.Tensor, name: str) -> None:
    """Print the XLA sharding spec for a single tensor. One-shot; callers should gate."""
    try:
        import torch_xla as _txla
        spec = _txla._XLAC._get_xla_sharding_spec(tensor)
    except Exception as e:  # noqa: BLE001
        spec = f"<error: {e}>"
    log.info("sharding[%s] shape=%s dtype=%s spec=%s",
             name, tuple(tensor.shape), tensor.dtype, spec)


_FIRST_INIT_STATE = True
_FIRST_SHARD_BATCH = True
_XLA_COMPILE_BASELINE = {"total": 0}


def _log_xla_compiles(step: int) -> None:
    """Emit a warning when XLA performs a new compile since last call. Reads CompileTime.TotalSamples."""
    import torch_xla.debug.metrics as met
    data = met.metric_data("CompileTime")
    if data is None:
        return
    total = data[0]
    delta = total - _XLA_COMPILE_BASELINE["total"]
    _XLA_COMPILE_BASELINE["total"] = total
    if delta == 0:
        return
    log.warning("xla_recompile step=%d compiles_since_last_log=%d total_compiles=%d",
                step, delta, total)


def _init_state(clf: CanViTForImageClassification, batch_size: int, device, mesh):
    global _FIRST_INIT_STATE
    state = clf.init_state(batch_size=batch_size, canvas_grid_size=CANVAS_GRID)
    state.canvas = state.canvas.to(device)
    state.recurrent_cls = state.recurrent_cls.to(device)
    if mesh is not None:
        xs.mark_sharding(state.canvas, mesh, ('data', None, None))
        xs.mark_sharding(state.recurrent_cls, mesh, ('data', None, None))
    if _FIRST_INIT_STATE:
        _FIRST_INIT_STATE = False
        _log_sharding_spec(state.canvas, "state.canvas")
        _log_sharding_spec(state.recurrent_cls, "state.recurrent_cls")
    return state


def _shard_batch(glimpses, labels, vp_centers, vp_scales, device, mesh):
    global _FIRST_SHARD_BATCH
    glimpses = glimpses.to(device)
    labels = labels.to(device)
    vp_centers = vp_centers.to(device)
    vp_scales = vp_scales.to(device)
    if mesh is not None:
        xs.mark_sharding(glimpses, mesh, (None, 'data', None, None, None))
        xs.mark_sharding(labels, mesh, ('data',))
    if _FIRST_SHARD_BATCH:
        _FIRST_SHARD_BATCH = False
        _log_sharding_spec(glimpses, "batch.glimpses")
        _log_sharding_spec(labels, "batch.labels")
    return glimpses, labels, vp_centers, vp_scales


def _init_comet(*, run_name: str | None, prev_key: str | None):
    if not os.environ.get("COMET_API_KEY"):
        return None
    import comet_ml
    cfg = comet_ml.ExperimentConfig(auto_metric_logging=False, name=run_name)
    kwargs: dict = {"experiment_config": cfg}
    if prev_key is not None:
        kwargs["experiment_key"] = prev_key
        log.info("Continuing Comet experiment: %s", prev_key)
    return comet_ml.start(**kwargs)


def _train_step(
    *,
    clf: CanViTForImageClassification,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    data_iter,
    grad_clip: float,
    label_smoothing: float,
    mesh: Mesh | None,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """One training step: N glimpses with per-chunk backward (truncated BPTT).

    Loss at every glimpse, normalized by N. Full BPTT when chunk_size >= N.
    """
    glimpses, labels, vp_centers, vp_scales = next(data_iter)
    glimpses, labels, vp_centers, vp_scales = _shard_batch(
        glimpses, labels, vp_centers, vp_scales, device, mesh)

    N, batch_size = glimpses.shape[:2]
    state = _init_state(clf, batch_size, device, mesh)

    chunk_loss = torch.zeros((), device=device)
    total_loss = torch.zeros((), device=device)
    correct_t0 = torch.zeros((), dtype=torch.long, device=device)
    loss_t0 = torch.zeros((), device=device)

    for g in range(N):
        vp = Viewpoint(centers=vp_centers[g], scales=vp_scales[g])
        logits, state = clf(glimpse=glimpses[g], state=state, viewpoint=vp)
        step_loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        chunk_loss = chunk_loss + step_loss
        total_loss = total_loss + step_loss.detach()

        if g == 0:
            correct_t0 = (logits.argmax(dim=-1) == labels).sum()
            loss_t0 = step_loss.detach()

        is_chunk_end = (g + 1) % chunk_size == 0
        is_last = g == N - 1

        if is_chunk_end or is_last:
            (chunk_loss / N).backward()
            if not is_last:
                # Truncated BPTT (chunk_size < N): dispatch each chunk's graph
                # early so memory = O(chunk_size), not O(N). When chunk_size == N
                # this branch is skipped and the end-of-step torch_xla.sync()
                # handles the whole fwd+bwd+optimizer graph in one shot.
                torch_xla.sync()
                state.canvas = state.canvas.detach()
                state.recurrent_cls = state.recurrent_cls.detach()
                chunk_loss = torch.zeros((), device=device)

    loss = total_loss / N
    n_correct = (logits.argmax(dim=-1) == labels).sum()
    loss_tN = step_loss.detach()

    grad_norm = torch.cat([p.grad.detach().float().flatten()
                           for p in clf.parameters() if p.grad is not None]).norm(2)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(clf.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    torch_xla.sync()

    return loss, n_correct, grad_norm, {
        'correct_t0': correct_t0, 'loss_t0': loss_t0, 'loss_tN': loss_tN,
    }


@torch.no_grad()
def _validate(
    *,
    clf: CanViTForImageClassification,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    n_val_steps: int,
    mesh: Mesh | None,
    policy_tag: str,
    exp,
    step: int,
    log_samples: bool,
) -> dict[str, float]:
    """Run full val set. Returns per-timestep accuracy and loss."""
    clf.eval()
    val_iter = iter(val_loader)
    total = 0
    t0_time = time.perf_counter()

    correct_per_t: list[torch.Tensor] | None = None
    loss_per_t_sum: list[torch.Tensor] | None = None
    sample_glimpse = sample_logits = sample_labels = None

    for i in range(n_val_steps):
        glimpses, labels, vp_centers, vp_scales = next(val_iter)

        if i == 0 and log_samples:
            sample_glimpse = glimpses[0, :8].clone()

        glimpses, labels, vp_centers, vp_scales = _shard_batch(
            glimpses, labels, vp_centers, vp_scales, device, mesh)

        N, B = glimpses.shape[:2]
        if correct_per_t is None:
            correct_per_t = [torch.zeros((), dtype=torch.long, device=device) for _ in range(N)]
            loss_per_t_sum = [torch.zeros((), device=device) for _ in range(N)]

        state = _init_state(clf, B, device, mesh)

        for g in range(N):
            vp = Viewpoint(centers=vp_centers[g], scales=vp_scales[g])
            logits, state = clf(glimpse=glimpses[g], state=state, viewpoint=vp)
            correct_per_t[g] += (logits.argmax(dim=-1) == labels).sum()
            loss_per_t_sum[g] += F.cross_entropy(logits, labels)

            if i == 0 and g == 0 and log_samples:
                sample_logits = logits[:8].float().cpu()
                sample_labels = labels[:8].cpu()

        total += B

    torch_xla.sync(wait=True)
    elapsed = time.perf_counter() - t0_time
    clf.train()

    assert correct_per_t is not None and loss_per_t_sum is not None
    N = len(correct_per_t)

    tag = f"_{policy_tag}" if policy_tag else ""
    result: dict[str, float] = {f"val/time{tag}_sec": elapsed}
    for g in range(N):
        result[f"val/accuracy{tag}_t{g}"] = correct_per_t[g].item() / total
        result[f"val/loss{tag}_t{g}"] = loss_per_t_sum[g].item() / n_val_steps

    if log_samples and exp and sample_glimpse is not None and sample_logits is not None:
        _log_val_samples(exp=exp, step=step, glimpses=sample_glimpse,
                         logits=sample_logits, labels=sample_labels,
                         imagenet_mean=IMAGENET_MEAN, imagenet_std=IMAGENET_STD)

    return result


def _run_validation(
    *,
    clf: CanViTForImageClassification,
    device: torch.device,
    val_loaders: list[ValLoader],
    n_val_steps: int,
    mesh: Mesh | None,
    exp,
    step: int,
    label: str,
    log_samples: bool,
) -> float:
    """Run all val loaders, log results. Returns accuracy from the primary loader."""
    primary_acc = -1.0
    for vl in val_loaders:
        metrics = _validate(
            clf=clf, device=device, val_loader=vl.loader,
            n_val_steps=n_val_steps, mesh=mesh, policy_tag=vl.tag,
            exp=exp, step=step, log_samples=log_samples and vl.tag == "",
        )
        pfx = f"_{vl.tag}" if vl.tag else ""
        acc = metrics[f"val/accuracy{pfx}_t{vl.n_glimpses-1}"]
        curve = " → ".join(f"{metrics[f'val/accuracy{pfx}_t{g}']:.1%}" for g in range(vl.n_glimpses))
        sec = metrics[f"val/time{pfx}_sec"]
        log.info("  VAL %s (%s, N=%d) | acc %.1f%% | curve: [%s] | %.1fs",
                 label, vl.display, vl.n_glimpses, acc * 100, curve, sec)
        if exp:
            exp.log_metrics(metrics, step=step)
        if vl.tag == "":
            primary_acc = acc
    return primary_acc


def train(args: argparse.Namespace) -> float:
    train_start = time.perf_counter()
    log.info("train() entered [%.1fs since Python start]", time.perf_counter() - _PYTHON_START)
    _log_environment_diagnostics()

    # SPMD
    n_devices = xr.global_runtime_device_count()
    mesh = None
    if n_devices > 1:
        xr.use_spmd()
        mesh = Mesh(np.arange(n_devices), (n_devices,), ('data',))
        log.info("SPMD: %d devices, mesh=(%d,), partition='data'", n_devices, n_devices)
    else:
        log.info("No SPMD: only %d device; mark_sharding skipped", n_devices)

    device = torch_xla.device()
    N = args.n_glimpses
    steps_per_epoch = TRAIN_IMAGES // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    n_val_steps = args.val_steps if args.val_steps > 0 else math.ceil(VAL_IMAGES / args.batch_size)

    log.info("Device: %s", device)
    log.info("Plan: %d epochs × %d steps/epoch = %d total, N=%d glimpses, chunk=%d",
             args.epochs, steps_per_epoch, total_steps, N, args.chunk_size)

    t0 = time.perf_counter()
    train_loader = make_multi_glimpse_dataloader(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, n_glimpses=N, split="train",
        glimpse_size=args.glimpse_size, min_viewpoint_scale=args.min_viewpoint_scale,
        t0_full_scene=True,
    )

    eval_n = args.eval_n_glimpses if args.eval_n_glimpses > 0 else N
    val_loaders: list[ValLoader] = [
        ValLoader(
            make_multi_glimpse_dataloader(
                data_dir=args.data_dir, batch_size=args.batch_size,
                num_workers=args.num_workers, n_glimpses=eval_n, split="validation",
                glimpse_size=args.glimpse_size, min_viewpoint_scale=args.min_viewpoint_scale,
                t0_full_scene=True, viewpoint_policy="random",
            ),
            eval_n, "", "random",
        ),
    ]
    if args.eval_c2f:
        val_loaders.append(ValLoader(
            make_multi_glimpse_dataloader(
                data_dir=args.data_dir, batch_size=args.batch_size,
                num_workers=args.num_workers, n_glimpses=eval_n, split="validation",
                glimpse_size=args.glimpse_size, t0_full_scene=True, viewpoint_policy="c2f",
            ),
            eval_n, "c2f", "c2f",
        ))

    train_iter = iter(train_loader)
    log.info("DataLoaders ready (B=%d, %dw, eval_N=%d) [%.1fs]",
             args.batch_size, args.num_workers, eval_n, time.perf_counter() - t0)

    t0 = time.perf_counter()
    clf = load_classifier(device=device)
    optimizer = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log.info("Model + optimizer ready [%.1fs]", time.perf_counter() - t0)

    start_step = 0
    best_val_acc = -1.0
    prev_comet_key: str | None = None
    if args.init_from:
        t0 = time.perf_counter()
        state = load_checkpoint(args.init_from)
        apply_model_weights(state, clf, device)
        log.info("Init from step %s [%.1fs]. Training from step 0.", state.get('step', '?'), time.perf_counter() - t0)
    elif args.checkpoint_dir:
        t0 = time.perf_counter()
        start_step, best_val_acc, prev_comet_key = maybe_resume(
            checkpoint_dir=args.checkpoint_dir, clf=clf, optimizer=optimizer, device=device,
        )
        log.info("Resume [%.1fs] → step %d, best_val_acc=%.4f", time.perf_counter() - t0, start_step, best_val_acc)

    t0 = time.perf_counter()
    exp = _init_comet(run_name=args.run_name, prev_key=prev_comet_key)
    if exp:
        log.info("Comet [%.1fs]: %s (%s)", time.perf_counter() - t0, exp.get_key(), args.run_name or 'auto')
        exp.log_parameters({
            "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
            "label_smoothing": args.label_smoothing,
            "epochs": args.epochs, "total_steps": total_steps,
            "glimpse_size": args.glimpse_size, "n_params": sum(p.numel() for p in clf.parameters()),
            "num_workers": args.num_workers, "warmup_steps": args.warmup_steps,
            "start_step": start_step, "checkpoint_dir": args.checkpoint_dir,
            "n_glimpses": N, "chunk_size": args.chunk_size,
            "min_viewpoint_scale": args.min_viewpoint_scale,
            "eval_n_glimpses": eval_n, "early_stop_delta": args.early_stop_delta,
            "init_from": args.init_from or "",
            "sky_task_id": os.environ.get("SKYPILOT_TASK_ID", ""),
            "sky_user": os.environ.get("SKYPILOT_USER", ""),
        })
    else:
        log.info("Comet skipped — COMET_API_KEY not set")

    comet_key = exp.get_key() if exp else None

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(warmup_steps=args.warmup_steps, total_steps=total_steps))
    for _ in range(start_step):
        scheduler.step()

    step_fn = partial(_train_step, chunk_size=args.chunk_size)
    log.info("Compiling XLA graph (N=%d, chunk=%d)...", N, args.chunk_size)
    t0 = time.perf_counter()
    step_fn(clf=clf, optimizer=optimizer, scheduler=scheduler, device=device,
            data_iter=train_iter, grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing, mesh=mesh)
    torch_xla.sync(wait=True)
    log.info("Compiled [%.1fs]", time.perf_counter() - t0)

    def _ckpt(step: int, filename: str = "latest.pt") -> None:
        if args.checkpoint_dir:
            save_checkpoint(checkpoint_dir=args.checkpoint_dir, step=step, clf=clf,
                            optimizer=optimizer, best_val_acc=best_val_acc,
                            comet_key=comet_key, filename=filename,
                            sync_fn=lambda: torch_xla.sync(wait=True))

    if start_step == 0:
        _ckpt(0)
        _ckpt(0, filename="init.pt")

    if args.pre_training_val and start_step == 0:
        log.info("Pre-training validation (step 0)...")
        _run_validation(clf=clf, device=device, val_loaders=val_loaders,
                        n_val_steps=n_val_steps, mesh=mesh,
                        exp=exp, step=0, label="step 0", log_samples=True)

    startup_sec = time.perf_counter() - train_start
    log.info("Training from step %d to %d [startup %.1fs]", start_step, total_steps, startup_sec)

    loss_sum = torch.zeros((), device=device)
    correct_sum = torch.zeros((), dtype=torch.long, device=device)
    grad_norm_sum = torch.zeros((), device=device)
    correct_t0_sum = torch.zeros((), dtype=torch.long, device=device)
    loss_t0_sum = torch.zeros((), device=device)
    loss_tN_sum = torch.zeros((), device=device)
    window_start = time.perf_counter()
    window_steps = 0
    window_samples = 0

    step = start_step
    for step in range(start_step, total_steps):
        loss, n_correct, grad_norm, extras = step_fn(
            clf=clf, optimizer=optimizer, scheduler=scheduler,
            device=device, data_iter=train_iter, grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing, mesh=mesh,
        )
        loss_sum += loss.detach()
        correct_sum += n_correct.detach()
        grad_norm_sum += grad_norm.detach()
        correct_t0_sum += extras['correct_t0'].detach()
        loss_t0_sum += extras['loss_t0']
        loss_tN_sum += extras['loss_tN']
        window_steps += 1
        window_samples += args.batch_size

        # Periodic logging
        if step % args.log_every == 0 and step > start_step:
            torch_xla.sync(wait=True)
            _log_xla_compiles(step)
            elapsed = time.perf_counter() - window_start
            sc_per_sec = window_samples / elapsed
            metrics = {
                "loss": (loss_sum / window_steps).item(),
                "accuracy": correct_sum.item() / window_samples,
                "grad_norm": (grad_norm_sum / window_steps).item(),
                "scenes_per_sec": sc_per_sec,
                "glimpses_per_sec": sc_per_sec * N,
                "ms_per_step": elapsed / window_steps * 1000,
                "lr": scheduler.get_last_lr()[0],
                "accuracy_t0": correct_t0_sum.item() / window_samples,
                "loss_t0": (loss_t0_sum / window_steps).item(),
                f"loss_t{N-1}": (loss_tN_sum / window_steps).item(),
            }
            epoch = step / steps_per_epoch
            log.info("step %6d ep %.2f | loss %.4f (t0=%.3f t%d=%.3f) | "
                     "acc t0=%.1f%% t%d=%.1f%% | gnorm %.2e | lr %.2e | %.0f sc/s",
                     step, epoch, metrics['loss'], metrics['loss_t0'],
                     N-1, metrics[f'loss_t{N-1}'],
                     metrics['accuracy_t0'] * 100, N-1, metrics['accuracy'] * 100,
                     metrics['grad_norm'], metrics['lr'], metrics['scenes_per_sec'])
            if exp:
                exp.log_metrics(metrics, step=step)
            loss_sum.zero_()
            correct_sum.zero_()
            grad_norm_sum.zero_()
            correct_t0_sum.zero_()
            loss_t0_sum.zero_()
            loss_tN_sum.zero_()
            window_start = time.perf_counter()
            window_steps = 0
            window_samples = 0

        # Periodic checkpoint
        if args.checkpoint_every > 0 and step > 0 and step % args.checkpoint_every == 0:
            _ckpt(step)

        # Epoch boundary: validate + checkpoint + early stop
        if step > 0 and step % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            val_acc = _run_validation(
                clf=clf, device=device, val_loaders=val_loaders,
                n_val_steps=n_val_steps, mesh=mesh,
                exp=exp, step=step, label=f"ep {epoch}", log_samples=True,
            )
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                log.info("  New best: %.1f%%", best_val_acc * 100)
            _ckpt(step)
            if is_best:
                _ckpt(step, filename="best.pt")
            if should_early_stop(val_acc=val_acc, best_val_acc=best_val_acc, delta=args.early_stop_delta):
                log.info("  EARLY STOP: val %.1f%% < best %.1f%% by %.1fpp (threshold %.1fpp)",
                         val_acc * 100, best_val_acc * 100,
                         (best_val_acc - val_acc) * 100, args.early_stop_delta * 100)
                break

    torch_xla.sync(wait=True)
    final_step = step + 1
    val_acc = _run_validation(
        clf=clf, device=device, val_loaders=val_loaders,
        n_val_steps=n_val_steps, mesh=mesh,
        exp=exp, step=final_step, label="final", log_samples=True,
    )
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        log.info("  New best: %.1f%%", best_val_acc * 100)
        _ckpt(final_step, filename="best.pt")
    _ckpt(final_step)

    if exp:
        exp.end()
    log.info("Done. best_val_acc=%.1f%%", best_val_acc * 100)
    return best_val_acc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--glimpse-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--lr", type=float, default=2.5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=25000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--n-glimpses", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument("--min-viewpoint-scale", type=float, default=0.05)
    p.add_argument("--eval-n-glimpses", type=int, default=0,
                   help="N for eval. 0 = same as --n-glimpses.")
    p.add_argument("--eval-c2f", action="store_true")
    p.add_argument("--early-stop-delta", type=float, default=None)
    p.add_argument("--val-steps", type=int, default=0)
    p.add_argument("--pre-training-val", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--init-from", default=None,
                   help="Load model weights only (no optimizer/step) from checkpoint.")
    p.add_argument("--run-name", default=None)
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
