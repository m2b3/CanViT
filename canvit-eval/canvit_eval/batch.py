"""Sequential single-GPU batch eval.

Outputs land in task-specific subdirs (ade20k_seg/, ade20k_seg_ablations/,
in1k_clf_{frozen,finetuned}/, recon/). Filenames carry a UTC timestamp;
--skip-existing matches on the structural identity (task, model, policy,
scene, grid, run_idx), so reruns resume cleanly across invocations.

Run `--help` for flags.
"""

import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import tyro
from canvit_pytorch import resolve_canvit_repo
from canvit_pytorch.checkpoints import (
    ABLATION_CHECKPOINTS,
    ABLATION_MODEL_SHORTS,
    PRETRAIN_CHECKPOINTS,
    ade20k_dinov3_probe_name,
    ade20k_probe_repo,
)

from canvit_eval.config import DINOV3_VITB_REPO, DINOV3_VITS_REPO
from canvit_eval.policies import IN1K_POLICIES, PolicyName, is_power_of_two

log = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


ALL_POLICIES: list[PolicyName] = list(get_args(PolicyName))

# No RNG at any point in the episode: bit-identical across runs given a fixed
# probe and dataloader order. Trimmed to n_runs=1.
DETERMINISTIC: frozenset[PolicyName] = frozenset({"repeated_full_scene", "entropy_coarse_to_fine"})


def _n_runs_for(policy: PolicyName, n_runs: int) -> int:
    return 1 if policy in DETERMINISTIC else n_runs


def _policy_runs_on_grid(policy: PolicyName, canvas_grid: int) -> bool:
    # entropy_coarse_to_fine partitions the canvas into 2x2 / 4x4 tiles; only
    # power-of-two grids align (otherwise _build_tile_masks asserts).
    if policy == "entropy_coarse_to_fine":
        return is_power_of_two(canvas_grid)
    return True


# (scene_size, canvas_grid, batch_size) for CanViT multi-timestep evals.
ADE20K_RESOLUTIONS: list[tuple[int, int, int]] = [
    (512, 32, 32),
    (512, 64, 8),
    (1024, 64, 8),
]

# (scene_size, canvas_grid) for CanViT single-glimpse (t=0) probes.
CANVAS_GRIDS: list[tuple[int, int]] = [
    (512, 8), (512, 16), (512, 32), (1024, 64),
]

# --include-extra-grids: append to CANVAS_GRIDS / ADE20K_RESOLUTIONS. Lists are
# disjoint — c8 and c16 already have t=0 data in CANVAS_GRIDS so they don't
# reappear in EXTRA_CANVAS_GRIDS (would produce duplicate output paths).
EXTRA_CANVAS_GRIDS: list[tuple[int, int]] = [
    (512, 9), (512, 10), (512, 12), (512, 24),
]
EXTRA_ADE20K_RESOLUTIONS: list[tuple[int, int, int]] = [
    (512, 8, 32),
    (512, 9, 32),
    (512, 10, 32),
    (512, 12, 32),
    (512, 16, 32),
    (512, 24, 32),
]

# Fusion uses the c=32 CLS standardizer (the only one initialised by pretraining);
# runtime canvas_grid can still vary. See FUSION_CANVAS_GRID in tasks/in1k_clf.py.
IN1K_RESOLUTIONS: list[tuple[int, int, int]] = [
    (512, 32, 64),
]
EXTRA_IN1K_RESOLUTIONS: list[tuple[int, int, int]] = [
    (512, 8, 64),
    (512, 16, 64),
    (512, 64, 8),
]

DINOV3_VARIANTS: dict[str, str] = {
    "dv3b": DINOV3_VITB_REPO,
    "dv3s": DINOV3_VITS_REPO,
}
DINOV3_RESOLUTIONS: list[int] = [128, 144, 160, 192, 256, 384, 512]

# {scene_size: batch_size} derived from the matrix; raises KeyError on unknown scene.
_BATCH_SIZE_BY_SCENE: dict[int, int] = {
    scene: bs for scene, _, bs in ADE20K_RESOLUTIONS + EXTRA_ADE20K_RESOLUTIONS
}


def _probe_repo(scene: int, grid: int) -> str:
    return ade20k_probe_repo("in21k", scene=scene, grid=grid)


TaskName = Literal["ade20k-seg", "ade20k-seg-ablations", "ade20k-seg-pretrain", "in1k-clf", "in1k-clf-ablations", "in1k-clf-pretrain", "recon"]
# Paper-v1 matrix; ade20k-seg-ablations is opt-in so existing invocations
# don't silently grow.
DEFAULT_TASKS: list[TaskName] = ["ade20k-seg", "in1k-clf", "recon"]


# Matches the UTC timestamp `_utc_timestamp` writes into every output filename.
_TS_RE = re.compile(r"\d{8}T\d{6}Z")


@dataclass(frozen=True)
class EvalJob:
    task: TaskName
    args: list[str]
    output: Path
    output_stem: str
    model: str
    policy: PolicyName | None = None
    scene_size: int | None = None
    canvas_grid: int | None = None
    input_px: int | None = None
    run_idx: int = 0

    def already_done(self) -> bool:
        """Glob match on this job's filename with the timestamp replaced by `*`."""
        pattern = _TS_RE.sub("*", self.output.name)
        return any(self.output.parent.glob(pattern))

    def describe(self) -> str:
        parts = [self.task, f"model={self.model}"]
        if self.policy:
            parts.append(f"policy={self.policy}")
        if self.scene_size:
            parts.append(f"s={self.scene_size}")
        if self.canvas_grid:
            parts.append(f"c={self.canvas_grid}")
        if self.input_px:
            parts.append(f"input={self.input_px}px")
        parts.append(f"r={self.run_idx}")
        return " ".join(parts)


def _ade20k_seg_jobs(
    out_dir: Path, *, n_runs: int, n_timesteps: int, ts: str,
    ade20k_res: list[tuple[int, int, int]], canvas_grids: list[tuple[int, int]],
) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    d = out_dir / "ade20k_seg"
    batch_size_by_scene = {scene: bs for scene, _, bs in ade20k_res}

    # CanViT multi-timestep policy evals.
    for scene, grid, bs in ade20k_res:
        probe = _probe_repo(scene, grid)
        for policy in ALL_POLICIES:
            if not _policy_runs_on_grid(policy, grid):
                continue
            for run in range(_n_runs_for(policy, n_runs)):
                stem = f"{policy}_s{scene}_c{grid}"
                out = d / f"{stem}_{ts}_r{run}.pt"
                jobs.append(EvalJob(
                    task="ade20k-seg",
                    args=["ade20k-seg-canvit", "--probe-repo", probe,
                          "--episode.policy", policy, "--episode.n-timesteps", str(n_timesteps),
                          "--episode.canvas-grid", str(grid), "--scene-size", str(scene),
                          "--batch-size", str(bs), "--output", str(out)],
                    output=out, output_stem=f"{stem}_",
                    model="canvit", policy=policy,
                    scene_size=scene, canvas_grid=grid, run_idx=run,
                ))

    # DINOv3 baselines: single passive forward, deterministic.
    for variant, teacher_repo in DINOV3_VARIANTS.items():
        for res in DINOV3_RESOLUTIONS:
            stem = f"{variant}_{res}px"
            out = d / f"{stem}_{ts}.pt"
            jobs.append(EvalJob(
                task="ade20k-seg",
                args=["ade20k-seg-dinov3",
                      "--probe-repo", resolve_canvit_repo(ade20k_dinov3_probe_name(variant, resolution=res)),
                      "--teacher-repo", teacher_repo,
                      "--eval-resolution", str(res), "--output", str(out)],
                output=out, output_stem=f"{stem}_",
                model=f"dinov3-{variant[-1]}",
                input_px=res, canvas_grid=res // 16,
            ))

    # CanViT single-glimpse probes (t=0 full-scene; deterministic).
    for scene, grid in canvas_grids:
        bs = batch_size_by_scene[scene]
        stem = f"canvit_s{scene}_c{grid}"
        out = d / f"{stem}_{ts}.pt"
        jobs.append(EvalJob(
            task="ade20k-seg",
            args=["ade20k-seg-canvit", "--probe-repo", _probe_repo(scene, grid),
                  "--episode.policy", "coarse_to_fine", "--episode.n-timesteps", "1",
                  "--episode.canvas-grid", str(grid), "--scene-size", str(scene),
                  "--batch-size", str(bs), "--output", str(out)],
            output=out, output_stem=f"{stem}_",
            model="canvit", policy="coarse_to_fine",
            scene_size=scene, canvas_grid=grid, input_px=128,
        ))

    return jobs


def _ablation_seg_jobs(
    out_dir: Path, *, n_runs: int, n_timesteps: int, ts: str,
) -> list[EvalJob]:
    """Paper-protocol ADE20K seg for every pretraining-ablation checkpoint.

    Same policy matrix as the flagship at the (512, 32) config. The eval CLI
    defaults `--episode.model-repo` to the flagship and most ablation variants
    share its canvas_dim, so model and probe are both derived from the same
    ABLATION_CHECKPOINTS entry here — never rely on the default.
    """
    jobs: list[EvalJob] = []
    d = out_dir / "ade20k_seg_ablations"
    scene, grid, bs = next(r for r in ADE20K_RESOLUTIONS if r[:2] == (512, 32))
    for slug, model_repo in ABLATION_CHECKPOINTS.items():
        probe = ade20k_probe_repo(ABLATION_MODEL_SHORTS[model_repo], scene=scene, grid=grid)
        for policy in ALL_POLICIES:
            if not _policy_runs_on_grid(policy, grid):
                continue
            for run in range(_n_runs_for(policy, n_runs)):
                stem = f"abl-{slug}_{policy}_s{scene}_c{grid}"
                out = d / f"{stem}_{ts}_r{run}.pt"
                jobs.append(EvalJob(
                    task="ade20k-seg-ablations",
                    args=["ade20k-seg-canvit", "--probe-repo", probe,
                          "--episode.model-repo", model_repo,
                          "--episode.policy", policy, "--episode.n-timesteps", str(n_timesteps),
                          "--episode.canvas-grid", str(grid), "--scene-size", str(scene),
                          "--batch-size", str(bs), "--output", str(out)],
                    output=out, output_stem=f"{stem}_",
                    model=f"abl-{slug}", policy=policy,
                    scene_size=scene, canvas_grid=grid, run_idx=run,
                ))
    return jobs


def _in1k_clf_jobs(
    out_dir: Path, *, n_runs: int, n_timesteps: int, ts: str,
    mode: Literal["frozen", "finetuned"],
    resolutions: list[tuple[int, int, int]],
) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    d = out_dir / f"in1k_clf_{mode}"
    for scene, grid, bs in resolutions:
        for policy in IN1K_POLICIES:
            for run in range(_n_runs_for(policy, n_runs)):
                stem = f"in1k_{policy}_s{scene}_c{grid}"
                out = d / f"{stem}_{ts}_r{run}.pt"
                jobs.append(EvalJob(
                    task="in1k-clf",
                    args=["in1k-clf", "--mode", mode,
                          "--scene-size", str(scene),
                          "--batch-size", str(bs),
                          "--episode.policy", policy,
                          "--episode.canvas-grid", str(grid),
                          "--episode.n-timesteps", str(n_timesteps),
                          "--output", str(out)],
                    output=out, output_stem=f"{stem}_",
                    model=f"canvit-{mode}", policy=policy,
                    scene_size=scene, canvas_grid=grid, run_idx=run,
                ))
    return jobs


def _in1k_clf_ablation_jobs(
    out_dir: Path, *, n_runs: int, n_timesteps: int, ts: str,
) -> list[EvalJob]:
    """Frozen IN1k classification for every pretraining-ablation checkpoint, C2F.

    The frozen head is algebraically fused at load time from the checkpoint's own
    CLS projection + the SHARED DINOv3 IN1k probe (no per-model probe training,
    unlike ADE20K). Only the model differs per variant, so we pin
    --episode.model-repo explicitly (the CLI defaults it to the flagship) and
    leave --probe-repo at its default shared DINOv3 probe.
    """
    jobs: list[EvalJob] = []
    d = out_dir / "in1k_clf_ablations"
    scene, grid, bs = IN1K_RESOLUTIONS[0]  # (512, 32, 64) — the paper's frozen config
    policy: PolicyName = "coarse_to_fine"
    for slug, model_repo in ABLATION_CHECKPOINTS.items():
        for run in range(_n_runs_for(policy, n_runs)):
            stem = f"abl-{slug}_{policy}_s{scene}_c{grid}"
            out = d / f"{stem}_{ts}_r{run}.pt"
            jobs.append(EvalJob(
                task="in1k-clf-ablations",
                args=["in1k-clf", "--mode", "frozen",
                      "--episode.model-repo", model_repo,
                      "--scene-size", str(scene),
                      "--batch-size", str(bs),
                      "--episode.policy", policy,
                      "--episode.canvas-grid", str(grid),
                      "--episode.n-timesteps", str(n_timesteps),
                      "--output", str(out)],
                output=out, output_stem=f"{stem}_",
                model=f"abl-{slug}", policy=policy,
                scene_size=scene, canvas_grid=grid, run_idx=run,
            ))
    return jobs


def _in1k_clf_pretrain_jobs(
    out_dir: Path, *, n_runs: int, n_timesteps: int, ts: str,
    max_batch_size: int | None,
) -> list[EvalJob]:
    """Frozen IN1k classification for each pretraining-dataset checkpoint
    (IN21k flagship vs IN1k-only), for the IN1k-vs-IN21k dataset-scale comparison.

    Same frozen protocol as the flagship (scene 512, canvas 32, fused SHARED
    DINOv3 probe); only the model differs, pinned per PRETRAIN_CHECKPOINTS entry.
    Both checkpoints share the flagship canvas_dim, so the shared probe fuses for
    each. Machine-agnostic: runs anywhere IMAGENET_VAL resolves — an H100 fits
    bs=64; on a 24 GB card pass --max-batch-size (e.g. 16), since top-1 is
    batch-invariant.
    """
    jobs: list[EvalJob] = []
    d = out_dir / "in1k_clf_pretrain"
    scene, grid, bs = _cap_batch_sizes([IN1K_RESOLUTIONS[0]], max_batch_size)[0]
    # Only the IN21k-flagship vs IN1k-only dataset-scale comparison. sa1b (also in
    # PRETRAIN_CHECKPOINTS) is out of scope — abandoned continued-pretrain at a
    # different scene resolution.
    for short in ("in21k", "in1k"):
        model_repo = PRETRAIN_CHECKPOINTS[short]
        for policy in IN1K_POLICIES:
            for run in range(_n_runs_for(policy, n_runs)):
                stem = f"pretrain-{short}_{policy}_s{scene}_c{grid}"
                out = d / f"{stem}_{ts}_r{run}.pt"
                jobs.append(EvalJob(
                    task="in1k-clf-pretrain",
                    args=["in1k-clf", "--mode", "frozen",
                          "--episode.model-repo", model_repo,
                          "--scene-size", str(scene),
                          "--batch-size", str(bs),
                          "--episode.policy", policy,
                          "--episode.canvas-grid", str(grid),
                          "--episode.n-timesteps", str(n_timesteps),
                          "--output", str(out)],
                    output=out, output_stem=f"{stem}_",
                    model=f"pretrain-{short}", policy=policy,
                    scene_size=scene, canvas_grid=grid, run_idx=run,
                ))
    return jobs


# Scene-512 grids we train IN1k-only ADE20K probes for (c32 native, c64 headline).
_SEG_PRETRAIN_RES: list[tuple[int, int, int]] = [(512, 32, 32), (512, 64, 8)]


def _ade20k_seg_pretrain_jobs(
    out_dir: Path, *, n_runs: int, n_timesteps: int, ts: str,
    max_batch_size: int | None,
) -> list[EvalJob]:
    """ADE20K seg for each pretraining-dataset checkpoint (IN21k flagship vs IN1k-only),
    at scene 512, canvas 32 and 64, full policy matrix.

    Unlike clf there is no shared fused head: each model needs its OWN ADE20K probe. Both
    the model AND the probe are pinned per PRETRAIN_CHECKPOINTS entry — model = the repo,
    probe = probe-ade20k-40k-s512-c{grid}-{short}. The in21k arm reuses the existing
    flagship probes (so it doubles as a seg faithfulness check vs the paper); the in1k arm
    uses the probes trained for step-1837056. sa1b excluded (out of scope).
    """
    jobs: list[EvalJob] = []
    d = out_dir / "ade20k_seg_pretrain"
    for scene, grid, bs in _cap_batch_sizes(_SEG_PRETRAIN_RES, max_batch_size):
        for short in ("in21k", "in1k"):
            model_repo = PRETRAIN_CHECKPOINTS[short]
            probe = ade20k_probe_repo(short, scene=scene, grid=grid)
            for policy in ALL_POLICIES:
                if not _policy_runs_on_grid(policy, grid):
                    continue
                for run in range(_n_runs_for(policy, n_runs)):
                    stem = f"pretrain-{short}_{policy}_s{scene}_c{grid}"
                    out = d / f"{stem}_{ts}_r{run}.pt"
                    jobs.append(EvalJob(
                        task="ade20k-seg-pretrain",
                        args=["ade20k-seg-canvit", "--probe-repo", probe,
                              "--episode.model-repo", model_repo,
                              "--episode.policy", policy, "--episode.n-timesteps", str(n_timesteps),
                              "--episode.canvas-grid", str(grid), "--scene-size", str(scene),
                              "--batch-size", str(bs), "--output", str(out)],
                        output=out, output_stem=f"{stem}_",
                        model=f"pretrain-{short}", policy=policy,
                        scene_size=scene, canvas_grid=grid, run_idx=run,
                    ))
    return jobs


def _recon_jobs(out_dir: Path, *, n_runs: int, ts: str) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    d = out_dir / "recon"
    for slug, repo in ABLATION_CHECKPOINTS.items():
        for run in range(n_runs):
            stem = f"recon_{slug}"
            out = d / f"{stem}_{ts}_r{run}.pt"
            jobs.append(EvalJob(
                task="recon",
                args=["reconstruction", "--model-repo", repo, "--output", str(out)],
                output=out, output_stem=f"{stem}_",
                model=slug, run_idx=run,
            ))
    return jobs


def _cap_batch_sizes(
    resolutions: list[tuple[int, int, int]], max_batch_size: int | None,
) -> list[tuple[int, int, int]]:
    if max_batch_size is None:
        return resolutions
    if max_batch_size <= 0:
        raise ValueError("--max-batch-size must be positive")
    return [(scene, grid, min(bs, max_batch_size)) for scene, grid, bs in resolutions]


def build_eval_matrix(
    out_dir: Path, *,
    n_runs: int,
    n_timesteps: int,
    tasks: list[TaskName],
    include_extra_grids: bool = False,
    max_batch_size: int | None = None,
) -> list[EvalJob]:
    """Generate all eval jobs. Pure — no side effects except the timestamp."""
    ts = _utc_timestamp()
    ade20k_res = ADE20K_RESOLUTIONS + (EXTRA_ADE20K_RESOLUTIONS if include_extra_grids else [])
    ade20k_res = _cap_batch_sizes(ade20k_res, max_batch_size)
    canvas_grids = CANVAS_GRIDS + (EXTRA_CANVAS_GRIDS if include_extra_grids else [])

    jobs: list[EvalJob] = []
    if "ade20k-seg" in tasks:
        jobs.extend(_ade20k_seg_jobs(
            out_dir, n_runs=n_runs, n_timesteps=n_timesteps, ts=ts,
            ade20k_res=ade20k_res, canvas_grids=canvas_grids,
        ))
    if "ade20k-seg-ablations" in tasks:
        jobs.extend(_ablation_seg_jobs(out_dir, n_runs=n_runs, n_timesteps=n_timesteps, ts=ts))
    if "in1k-clf-ablations" in tasks:
        jobs.extend(_in1k_clf_ablation_jobs(out_dir, n_runs=n_runs, n_timesteps=n_timesteps, ts=ts))
    if "in1k-clf-pretrain" in tasks:
        jobs.extend(_in1k_clf_pretrain_jobs(
            out_dir, n_runs=n_runs, n_timesteps=n_timesteps, ts=ts,
            max_batch_size=max_batch_size))
    if "ade20k-seg-pretrain" in tasks:
        jobs.extend(_ade20k_seg_pretrain_jobs(
            out_dir, n_runs=n_runs, n_timesteps=n_timesteps, ts=ts,
            max_batch_size=max_batch_size))
    if "in1k-clf" in tasks:
        # Extra grids only expand frozen mode; finetuned weights were specialised
        # at one (scene, grid) and off-grid inference is a separate question.
        frozen_res = IN1K_RESOLUTIONS + (EXTRA_IN1K_RESOLUTIONS if include_extra_grids else [])
        frozen_res = _cap_batch_sizes(frozen_res, max_batch_size)
        finetuned_res = _cap_batch_sizes(IN1K_RESOLUTIONS, max_batch_size)
        jobs.extend(_in1k_clf_jobs(out_dir, n_runs=n_runs, n_timesteps=n_timesteps, ts=ts,
                                    mode="frozen", resolutions=frozen_res))
        jobs.extend(_in1k_clf_jobs(out_dir, n_runs=n_runs, n_timesteps=n_timesteps, ts=ts,
                                    mode="finetuned", resolutions=finetuned_res))
    if "recon" in tasks:
        jobs.extend(_recon_jobs(out_dir, n_runs=n_runs, ts=ts))

    # Breadth-first across run_idx: every cell runs once at r=0 before any r=1,
    # so an interrupted batch still yields n=1 per cell. Filenames are pure
    # functions of (structural fields, invocation ts), so reordering here
    # doesn't affect filenames, skip-existing, or any consumer.
    jobs.sort(key=lambda j: (
        j.run_idx, j.task, j.model,
        j.scene_size or 0, j.canvas_grid or 0, j.input_px or 0,
        j.policy or "",
    ))
    return jobs


def filter_jobs(
    jobs: list[EvalJob], *,
    policies: list[str] | None = None,
    grids: list[int] | None = None,
) -> list[EvalJob]:
    """Drop jobs whose policy/canvas_grid is None or doesn't match the filter."""
    out = jobs
    if policies:
        policy_set = set(policies)
        out = [j for j in out if j.policy is not None and j.policy in policy_set]
    if grids:
        grid_set = set(grids)
        out = [j for j in out if j.canvas_grid is not None and j.canvas_grid in grid_set]
    return out


@dataclass
class Args:
    out_dir: Path = Path("results")
    n_runs: int = 5
    n_timesteps: int = 21
    tasks: list[TaskName] = field(default_factory=lambda: list(DEFAULT_TASKS))
    dry_run: bool = False
    skip_existing: bool = False
    """Skip jobs whose structural output already exists (timestamp-agnostic glob match)."""
    include_extra_grids: bool = False
    """Include extra canvas grids beyond the paper-v1 set.
       ADE20K: adds c8..c24 at s=512. IN1k frozen: adds c8/c16/c64 at matched scene sizes
       (finetuned stays at c=32 — it was specialized there)."""
    max_batch_size: int | None = None
    """Clamp every generated job's batch size to this value. Useful on 8 GB GPUs."""
    policies: list[str] = field(default_factory=list)
    """Filter to these policies (empty = all). Applies to jobs that carry a policy field."""
    grids: list[int] = field(default_factory=list)
    """Filter to these canvas grids (empty = all). For DINOv3: grid = input_px // 16."""
    shard: str = ""
    """Run only one shard of the (sorted, filtered) job list: "k/N", 0-indexed,
       strided as jobs[k::N]. For a SLURM array, pass "$SLURM_ARRAY_TASK_ID/<count>".
       Applied before --skip-existing so the partition is stable across reruns."""


def _apply_shard(jobs: list[EvalJob], spec: str) -> list[EvalJob]:
    k_str, _, n_str = spec.partition("/")
    k, n = int(k_str), int(n_str)
    assert n >= 1 and 0 <= k < n, f"bad --shard {spec!r}: need 0 <= k < N, N >= 1"
    return jobs[k::n]


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    jobs = build_eval_matrix(
        args.out_dir,
        n_runs=args.n_runs, n_timesteps=args.n_timesteps, tasks=args.tasks,
        include_extra_grids=args.include_extra_grids,
        max_batch_size=args.max_batch_size,
    )
    jobs = filter_jobs(jobs, policies=args.policies or None, grids=args.grids or None)
    if args.shard:
        jobs = _apply_shard(jobs, args.shard)
        log.info("shard %s -> %d jobs", args.shard, len(jobs))

    n_total = len(jobs)
    if args.skip_existing:
        jobs = [j for j in jobs if not j.already_done()]
    n_skipped = n_total - len(jobs)

    by_task: dict[str, int] = {}
    for j in jobs:
        by_task[j.task] = by_task.get(j.task, 0) + 1
    log.info("%d jobs to run (%d skipped as already done)%s",
             len(jobs), n_skipped,
             ": " + ", ".join(f"{k}={v}" for k, v in sorted(by_task.items())) if by_task else "")

    if args.dry_run:
        for job in jobs:
            print(" ".join(["uv", "run", "python", "-m", "canvit_eval"] + job.args))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    total_elapsed = 0.0
    for i, job in enumerate(jobs, start=1):
        t0 = time.monotonic()
        log.info("[%d/%d] RUN  %s", i, len(jobs), job.describe())
        result = subprocess.run([sys.executable, "-m", "canvit_eval"] + job.args)
        elapsed = time.monotonic() - t0
        total_elapsed += elapsed
        if result.returncode != 0:
            log.error("[%d/%d] FAIL (%.1fs, exit %d)  %s", i, len(jobs), elapsed, result.returncode, job.output.name)
            failed += 1
        else:
            log.info("[%d/%d] OK   (%.1fs)  %s", i, len(jobs), elapsed, job.output.name)
            done += 1

    log.info("DONE: %d/%d completed, %d failed, %.1f min total",
             done, len(jobs), failed, total_elapsed / 60)


if __name__ == "__main__":
    main(tyro.cli(Args))
