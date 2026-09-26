"""Driver for bench/pt/run.py.

Generates one subprocess per (model, device, scene, dtype, threads) cell,
optionally repeated across passes with shuffled order, pre-flight gated on
GPU/CPU idle. Run `--help` for flags.
"""

import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro

log = logging.getLogger("bench.matrix")

ModelName = Literal["canvit", "dinov3-vitb16", "dinov3-vits16"]
Device = Literal["cpu", "cuda"]
Dtype = Literal["fp32", "amp-bf16"]

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_PY = SCRIPT_DIR / "run.py"


def physical_cores() -> int:
    """Linux: distinct core_ids under /sys topology. Fallback: os.cpu_count()."""
    ids: set[str] = set()
    for p in Path("/sys/devices/system/cpu/").glob("cpu[0-9]*/topology/core_id"):
        try:
            ids.add(p.read_text().strip())
        except OSError:
            pass
    return len(ids) or (os.cpu_count() or 1)


def logical_cores() -> int:
    return os.cpu_count() or 1


@dataclass(frozen=True)
class IdleThresholds:
    max_gpu_util_pct: int = 5
    max_gpu_procs: int = 0
    max_load_1min: float = 2.0


def preflight(t: IdleThresholds = IdleThresholds()) -> None:
    """Raise RuntimeError if GPU or CPU is busy beyond thresholds."""
    # GPU
    try:
        util = int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True).strip())
        procs_csv = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True).strip()
        n_procs = len([line for line in procs_csv.splitlines() if line.strip()])
        log.info(f"GPU: util={util}% procs={n_procs}")
        if util > t.max_gpu_util_pct or n_procs > t.max_gpu_procs:
            raise RuntimeError(f"GPU busy (util={util}% procs={n_procs})")
    except FileNotFoundError:
        log.info("GPU: nvidia-smi not found (skipping)")

    # 1-min load average; not summed pcpu (latter is lifetime-average and
    # over-reports idle boxes with long-lived daemons).
    with open("/proc/loadavg") as f:
        load1 = float(f.read().split()[0])
    log.info(f"CPU load (1 min): {load1:.2f}")
    if load1 > t.max_load_1min:
        raise RuntimeError(f"CPU load average {load1} > {t.max_load_1min}")


@dataclass(frozen=True)
class BenchJob:
    model: ModelName
    device: Device
    scene_px: int
    dtype: Dtype
    num_threads: int  # CPU only; 0 = torch default
    compiled: bool
    pass_idx: int

    def describe(self) -> str:
        parts = [self.model, self.device, f"s={self.scene_px}", self.dtype]
        if self.device == "cpu":
            parts.append(f"threads={self.num_threads}")
        if self.compiled:
            parts.append("compiled")
        parts.append(f"pass={self.pass_idx}")
        return " ".join(parts)

    def args(self) -> list[str]:
        a = [
            "--model", self.model,
            "--device", self.device,
            "--scene-px", str(self.scene_px),
            "--dtype", self.dtype,
        ]
        if self.device == "cpu" and self.num_threads > 0:
            a += ["--num-threads", str(self.num_threads)]
        if self.compiled:
            a += ["--compiled"]
        return a


def build_matrix(
    *,
    models: list[ModelName],
    cpu_scenes: list[int],
    cpu_threads: list[int],
    cuda_scenes: list[int],
    cuda_dtypes: list[Dtype],
    cuda_compiled: bool,
    passes: int,
    seed: int = 42,
) -> list[BenchJob]:
    """Generate all (pass × cell) jobs, shuffled per pass."""
    jobs: list[BenchJob] = []

    # CPU: fp32 eager only. CUDA: dtypes from `cuda_dtypes`, compiled per `cuda_compiled`.
    cpu_cells: list[tuple[ModelName, int, int]] = [
        (m, s, t) for m in models for s in cpu_scenes for t in cpu_threads
    ]
    cuda_cells: list[tuple[ModelName, int, Dtype]] = [
        (m, s, d) for m in models for s in cuda_scenes for d in cuda_dtypes
    ]

    for p in range(passes):
        rng = random.Random(seed + p)
        cpu_p = list(cpu_cells)
        rng.shuffle(cpu_p)
        cuda_p = list(cuda_cells)
        rng.shuffle(cuda_p)
        for m, s, t in cpu_p:
            jobs.append(BenchJob(m, "cpu", s, "fp32", t, False, p))
        for m, s, d in cuda_p:
            jobs.append(BenchJob(m, "cuda", s, d, 0, cuda_compiled, p))
    return jobs


Profile = Literal["fast", "full"]

PROFILES: dict[Profile, dict] = {
    "fast": {"passes": 1, "time_budget_s": 10.0, "max_iters": 100, "warmup_iters": 1, "min_iters": 2},
    "full": {"passes": 3, "time_budget_s": 20.0, "max_iters": 500, "warmup_iters": 3, "min_iters": 5},
}


@dataclass
class Args:
    profile: Profile = "fast"
    """`fast` for a quick run; `full` for distributional CIs across multiple
    passes. Individual flags override the profile."""
    models: list[ModelName] = field(default_factory=lambda: ["canvit", "dinov3-vitb16", "dinov3-vits16"])
    cpu_scenes: list[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    cpu_threads: list[int] = field(default_factory=list)
    """Empty = auto {1, physical_cores}."""
    cuda_scenes: list[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048])
    cuda_dtypes: list[Dtype] = field(default_factory=lambda: ["fp32", "amp-bf16"])
    cuda_compiled: bool = True
    passes: int | None = None
    """Override profile passes count."""
    time_budget_s: float | None = None
    """Override profile time-budget per config."""
    max_iters: int | None = None
    """Override profile iteration cap per config."""
    min_iters: int | None = None
    """Override profile minimum iterations (floor that beats time_budget)."""
    warmup_iters: int | None = None
    """Override profile warmup iterations."""
    dry_run: bool = False
    skip_preflight: bool = False
    skip_gpu_for_cpu_jobs: bool = False


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    preset = PROFILES[args.profile]
    passes = args.passes if args.passes is not None else preset["passes"]
    time_budget_s = args.time_budget_s if args.time_budget_s is not None else preset["time_budget_s"]
    max_iters = args.max_iters if args.max_iters is not None else preset["max_iters"]
    min_iters = args.min_iters if args.min_iters is not None else preset["min_iters"]
    warmup_iters = args.warmup_iters if args.warmup_iters is not None else preset["warmup_iters"]
    log.info(f"profile={args.profile}: passes={passes} time_budget_s={time_budget_s} "
             f"max_iters={max_iters} min_iters={min_iters} warmup_iters={warmup_iters}")

    if not args.cpu_threads:
        phys = physical_cores()
        args.cpu_threads = sorted({1, phys})
        log.info(f"Auto cpu_threads = {args.cpu_threads} (physical={phys})")

    jobs = build_matrix(
        models=args.models,
        cpu_scenes=args.cpu_scenes, cpu_threads=args.cpu_threads,
        cuda_scenes=args.cuda_scenes, cuda_dtypes=args.cuda_dtypes,
        cuda_compiled=args.cuda_compiled, passes=passes,
    )
    log.info(f"{len(jobs)} jobs across {passes} pass(es).")

    if args.dry_run:
        for j in jobs:
            print(j.describe())
        return

    if not args.skip_preflight:
        thresholds = IdleThresholds()
        has_cuda = any(j.device == "cuda" for j in jobs)
        if not has_cuda or args.skip_gpu_for_cpu_jobs:
            thresholds = IdleThresholds(max_gpu_util_pct=100, max_gpu_procs=10_000)
        preflight(thresholds)

    # User-scoped Inductor cache: survives /tmp cleanup so long-gap re-runs
    # don't pay the full cold compile.
    cache_dir = Path.home() / ".cache" / "torch" / "inductor"
    cache_dir.mkdir(parents=True, exist_ok=True)
    child_env = {**os.environ, "TORCHINDUCTOR_CACHE_DIR": str(cache_dir)}
    log.info(f"TORCHINDUCTOR_CACHE_DIR = {cache_dir}")

    done = failed = 0
    total = time.monotonic()
    for i, j in enumerate(jobs, 1):
        t0 = time.monotonic()
        log.info(f"[{i}/{len(jobs)}] RUN  {j.describe()}")
        cmd = [sys.executable, str(RUN_PY)] + j.args() + [
            "--time-budget-s", str(time_budget_s),
            "--max-iters", str(max_iters),
            "--min-iters", str(min_iters),
            "--warmup-iters", str(warmup_iters),
        ]
        rc = subprocess.call(cmd, env=child_env)
        dt = time.monotonic() - t0
        if rc != 0:
            log.error(f"[{i}/{len(jobs)}] FAIL ({dt:.1f}s, exit {rc})")
            failed += 1
        else:
            log.info(f"[{i}/{len(jobs)}] OK   ({dt:.1f}s)")
            done += 1
    log.info(f"DONE: {done}/{len(jobs)} completed, {failed} failed, {(time.monotonic()-total)/60:.1f} min.")


if __name__ == "__main__":
    main(tyro.cli(Args))
