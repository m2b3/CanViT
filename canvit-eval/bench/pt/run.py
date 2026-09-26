"""Per-forward-pass latency at batch_size=1 with explicit device sync each iteration.

scene_px controls workload for both models:
  - DINOv3: input is scene_px square; (scene_px // 16)² patches.
  - CanViT: glimpse fixed at 128 px; canvas_grid = scene_px // 16
    (canvas spatial-token count matches DINOv3's patch count at the same scene).

Streams per-iteration timings to JSONL.
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Literal

import torch
import torch._inductor.config
import tyro

from canvit_pytorch.backbone import create_backbone
from canvit_pytorch.model.base import CanViT, CanViTConfig
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint
from canvit_pytorch.teacher import load_teacher

from canvit_eval.config import DINOV3_VITB_REPO, DINOV3_VITS_REPO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("bench")

DINOV3_REPOS = {
    "dinov3-vitb16": DINOV3_VITB_REPO,
    "dinov3-vits16": DINOV3_VITS_REPO,
}
CANVIT_GLIMPSE_PX = 128
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass(frozen=True)
class Args:
    model: Literal["canvit", "dinov3-vitb16", "dinov3-vits16"] = "canvit"
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    scene_px: int = 512
    """Scene resolution in pixels. Teacher gets this as input_px.
    CanViT gets canvas_grid = scene_px // 16, glimpse fixed at 128px."""
    compiled: bool = False
    combo_kernels: bool = False
    """Enable torch._inductor.config.combo_kernels (requires --compiled)."""
    dtype: Literal["fp32", "amp-bf16"] = "amp-bf16"
    batch_size: int = 1
    time_budget_s: float = 120.0
    """Measurement time budget in seconds (excludes model loading)."""
    max_iters: int = 500
    """Stop after this many iterations even if time budget not exhausted."""
    min_iters: int = 2
    """Floor on measurement iterations — runs at least this many even if the
    time budget is already exhausted. Guarantees a sample on very slow cells."""
    num_threads: int = 0
    """Number of CPU threads (0 = PyTorch default). Only relevant for --device cpu."""
    warmup_iters: int = 1
    """Warmup iterations before measurement. Iter 0 triggers torch.compile and
    primes caches; additional warmups add no information unless the environment
    is noisy."""


# Weights stay fp32; bf16 is applied via autocast at compute time.


@contextmanager
def _autocast(args: Args, device: torch.device) -> Iterator[None]:
    if args.dtype == "amp-bf16":
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            yield
    else:
        yield


def _run_id(args: Args, ts: str) -> str:
    c = "c" if args.compiled else "e"
    combo = "_combo" if args.combo_kernels else ""
    bs = f"_bs{args.batch_size}" if args.batch_size != 1 else ""
    dev = f"_{args.device}" if args.device != "cuda" else ""
    thr = f"_t{args.num_threads}" if args.num_threads > 0 else ""
    if args.model == "canvit":
        cg = args.scene_px // 16
        return f"canvit_{c}_{args.dtype}_{args.scene_px}px_cg{cg}{bs}{combo}{dev}{thr}_{ts}"
    return f"{args.model}_{c}_{args.dtype}_{args.scene_px}px{bs}{combo}{dev}{thr}_{ts}"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _reset_peak_mem(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _read_peak_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    if device.type == "mps":
        return round(torch.mps.current_allocated_memory() / 1e6, 1)
    return None


def _measure_streaming(
    fn: Callable[[], None],
    out_path: Path,
    meta: dict,
    time_budget_s: float,
    max_iters: int,
    min_iters: int,
    warmup_iters: int,
    device: torch.device,
) -> None:
    """Warmup (not budgeted), reset peak-mem, then time `fn` until budget exhausted.
    Streams per-iter rows to JSONL; emits a peak_mem row at the end."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps({"type": "meta", **meta}) + "\n")
        f.flush()

        for w in range(warmup_iters):
            _sync(device)
            t0 = time.perf_counter()
            fn()
            _sync(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            log.info("  warmup %d: %.1fms", w, elapsed_ms)
            row = {"type": "warmup", "i": w, "ms": round(elapsed_ms, 4)}
            f.write(json.dumps(row) + "\n")
            f.flush()

        _reset_peak_mem(device)
        i = 0
        wall_start = time.perf_counter()
        while True:
            _sync(device)
            t0 = time.perf_counter()
            fn()
            _sync(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            wall_s = time.perf_counter() - wall_start

            row = {"type": "iter", "i": i, "ms": round(elapsed_ms, 4), "wall_s": round(wall_s, 3)}
            f.write(json.dumps(row) + "\n")
            f.flush()

            if i <= 3 or i % 50 == 0:
                log.info("  iter %d: %.2fms (wall %.1fs)", i, elapsed_ms, wall_s)

            i += 1
            if i >= min_iters and (wall_s >= time_budget_s or i >= max_iters):
                break

        peak_mb = _read_peak_mb(device)
        if peak_mb is not None:
            log.info("  peak memory: %.1f MB", peak_mb)
            f.write(json.dumps({"type": "peak_mem", "peak_mem_mb": peak_mb}) + "\n")
            f.flush()

    log.info("  %d measured iterations in %.1fs -> %s", i, time.perf_counter() - wall_start, out_path)


def _build_dinov3(args: Args, device: torch.device) -> Callable[[], None]:
    repo = DINOV3_REPOS[args.model]
    log.info("Loading %s from %s...", args.model, repo)
    t0 = time.perf_counter()
    teacher = load_teacher(repo, device).to(dtype=torch.float32).eval()
    n_params = sum(p.numel() for p in teacher.parameters())
    log.info("  loaded in %.1fs, %.1fM params", time.perf_counter() - t0, n_params / 1e6)

    if args.compiled:
        log.info("  torch.compile...")
        t0 = time.perf_counter()
        # Compile the inner HF model directly: the bench entry point here is
        # forward_norm_features(), and both teacher.compile() and
        # torch.compile(teacher) only intercept .forward() — they would silently
        # no-op without any warning.
        teacher.model = torch.compile(teacher.model)  # type: ignore[assignment]
        log.info("  registered in %.1fs", time.perf_counter() - t0)

    x = torch.randn(args.batch_size, 3, args.scene_px, args.scene_px,
                     device=device, dtype=torch.float32)
    n_patches = (args.scene_px // 16) ** 2
    log.info("  input: %dpx -> %d patches", args.scene_px, n_patches)

    def fwd(x=x) -> None:
        teacher.forward_norm_features(x)

    return fwd


def _build_canvit(args: Args, device: torch.device) -> Callable[[], None]:
    log.info("Creating CanViT...")
    backbone = create_backbone("vitb16")
    model = CanViT(backbone=backbone, cfg=CanViTConfig())
    model = model.to(device=device, dtype=torch.float32).eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("  %.1fM params", n_params / 1e6)

    if args.compiled:
        log.info("  torch.compile...")
        t0 = time.perf_counter()
        model.compile()
        log.info("  registered in %.1fs", time.perf_counter() - t0)

    canvas_grid = args.scene_px // 16
    bs = args.batch_size
    image = torch.randn(bs, 3, CANVIT_GLIMPSE_PX, CANVIT_GLIMPSE_PX,
                         device=device, dtype=torch.float32)
    vp = Viewpoint.full_scene(batch_size=bs, device=device)
    glimpse = sample_at_viewpoint(spatial=image, viewpoint=vp, glimpse_size_px=CANVIT_GLIMPSE_PX)
    n_glimpse_patches = (CANVIT_GLIMPSE_PX // 16) ** 2
    n_canvas_patches = canvas_grid ** 2
    log.info("  glimpse: %dpx (%d patches), canvas: %dx%d (%d patches)",
             CANVIT_GLIMPSE_PX, n_glimpse_patches, canvas_grid, canvas_grid, n_canvas_patches)

    def run(glimpse=glimpse, vp=vp) -> None:
        state = model.init_state(batch_size=bs, canvas_grid_size=canvas_grid)
        model(glimpse=glimpse, state=state, viewpoint=vp)

    return run


def _device_info(device: torch.device) -> tuple[str, float | None]:
    if device.type == "cuda":
        return torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9
    if device.type == "mps":
        return "Apple Silicon (MPS)", None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip(), None
    except FileNotFoundError:
        pass
    return "CPU", None


def main() -> None:
    args = tyro.cli(Args)

    assert args.scene_px % 16 == 0, f"scene_px must be divisible by 16, got {args.scene_px}"
    if args.combo_kernels:
        assert args.compiled, "--combo-kernels requires --compiled"
        torch._inductor.config.combo_kernels = True
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        log.info("Set num_threads = %d", args.num_threads)

    device = torch.device(args.device)
    dev_name, dev_mem_gb = _device_info(device)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rid = _run_id(args, ts)

    log.info("Device: %s%s", dev_name, f" ({dev_mem_gb:.1f} GB)" if dev_mem_gb else "")
    log.info("torch: %s", torch.__version__)
    log.info("Run: %s", rid)
    log.info("Args: %s", args)

    builder = _build_canvit if args.model == "canvit" else _build_dinov3
    with torch.inference_mode():
        fwd = builder(args, device)

        meta = {
            "device_name": dev_name,
            "torch_version": torch.__version__,
            "num_threads_actual": torch.get_num_threads(),
            "run_id": rid,
            "timestamp": ts,
            **{k: v for k, v in args.__dict__.items()},
        }
        if dev_mem_gb is not None:
            meta["device_mem_gb"] = round(dev_mem_gb, 1)
        if args.model == "canvit":
            meta["canvas_grid"] = args.scene_px // 16
            meta["glimpse_px"] = CANVIT_GLIMPSE_PX

        out_path = RESULTS_DIR / f"bench_{rid}.jsonl"
        log.info(
            "Measuring (budget %.0fs, max_iters %d, min_iters %d)...",
            args.time_budget_s, args.max_iters, args.min_iters,
        )
        with _autocast(args, device):
            _measure_streaming(fwd, out_path, meta, args.time_budget_s, args.max_iters, args.min_iters, args.warmup_iters, device)

    log.info("Done.")


if __name__ == "__main__":
    main()
