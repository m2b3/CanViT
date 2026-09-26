"""The paper's evaluation matrix: every task configuration behind its ADE20K, ImageNet-1k and ablation results.

Jobs fall into groups, each saved under <out_dir>/<group>/ with file names the paper's figure and
table pipeline parses: <name>_<UTC timestamp>_r<run>.pt, without the run suffix for jobs that run
once by construction (DINOv3 baselines, single-glimpse CanViT). Stochastic policies run num_runs
times, deterministic ones once. Jobs are ordered breadth-first over runs, so an interrupted batch
still covers every configuration once.
"""

import dataclasses
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, assert_never

from canvit_pytorch.evaluate.config import NUM_GLIMPSES, EpisodeConfig
from canvit_pytorch.evaluate.processes import run_in_fresh_process
from canvit_pytorch.evaluate.tasks.ade20k_segmentation import ADE20kSegmentationCanViT, ADE20kSegmentationDINOv3
from canvit_pytorch.evaluate.tasks.imagenet_classification import (
    FineTuned,
    FrozenWithFusedProbe,
    ImageNetClassification,
)
from canvit_pytorch.evaluate.tasks.reconstruction import ABLATION_EPISODE, Reconstruction
from canvit_pytorch.hub import repos
from canvit_pytorch.policies import POLICIES, PolicyName
from canvit_pytorch.pretrain.ablations import ABLATIONS
from canvit_pytorch.teacher import DINOV3_PATCH_SIZE, DINOV3_REPOS

log = logging.getLogger(__name__)

Group = Literal[
    "ade20k_seg", "ade20k_seg_ablations", "ade20k_seg_pretrain",
    "in1k_clf_frozen", "in1k_clf_finetuned", "in1k_clf_ablations", "in1k_clf_pretrain",
    "recon",
]
DEFAULT_GROUPS: tuple[Group, ...] = ("ade20k_seg", "in1k_clf_frozen", "in1k_clf_finetuned", "recon")

EvaluationTask = ADE20kSegmentationCanViT | ADE20kSegmentationDINOv3 | ImageNetClassification | Reconstruction


class Setting(NamedTuple):
    scene_size_px: int
    canvas_grid_size: int
    batch_size: int


ADE20K_SETTINGS = [Setting(512, 32, 32), Setting(512, 64, 8), Setting(1024, 64, 8)]
ADE20K_SINGLE_GLIMPSE_SETTINGS = [Setting(512, 8, 32), Setting(512, 16, 32), Setting(512, 32, 32), Setting(1024, 64, 8)]
ADE20K_PRETRAIN_SETTINGS = [Setting(512, 32, 32), Setting(512, 64, 8)]
IN1K_SETTINGS = [Setting(512, 32, 64)]
# --include-extra-grids: the canvas grid sweep beyond the main settings, for frozen models only.
EXTRA_ADE20K_SETTINGS = [Setting(512, grid, 32) for grid in (8, 9, 10, 12, 16, 24)]
EXTRA_ADE20K_SINGLE_GLIMPSE_SETTINGS = [Setting(512, grid, 32) for grid in (9, 10, 12, 24)]
EXTRA_IN1K_SETTINGS = [Setting(512, 8, 64), Setting(512, 16, 64), Setting(512, 64, 8)]

DINOV3_INPUT_SIZES_PX = (128, 144, 160, 192, 256, 384, 512)

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
TIMESTAMP_PATTERN = re.compile(r"\d{8}T\d{6}Z")


@dataclass(frozen=True)
class EvaluationJob:
    group: Group
    run_index: int
    task: EvaluationTask

    @property
    def output(self) -> Path:
        return self.task.output

    @property
    def policy(self) -> PolicyName | None:
        match self.task:
            case ADE20kSegmentationDINOv3():
                return None
            case ADE20kSegmentationCanViT() | ImageNetClassification() | Reconstruction():
                return self.task.episode.policy

    @property
    def output_grid_size(self) -> int:
        """The canvas grid, or DINOv3's patch grid."""
        match self.task:
            case ADE20kSegmentationDINOv3():
                return self.task.input_size_px // DINOV3_PATCH_SIZE
            case ADE20kSegmentationCanViT() | ImageNetClassification() | Reconstruction():
                return self.task.episode.canvas_grid_size

    def already_done(self) -> bool:
        """Whether an output with this job's name exists, whatever its timestamp."""
        return any(self.output.parent.glob(TIMESTAMP_PATTERN.sub("*", self.output.name)))


def num_runs_of(policy: PolicyName, num_runs: int) -> int:
    return 1 if POLICIES[policy].deterministic else num_runs


def ade20k_policies(canvas_grid_size: int) -> list[PolicyName]:
    return [name for name, spec in POLICIES.items() if spec.supports_canvas_grid(canvas_grid_size)]


def in1k_policies() -> list[PolicyName]:
    return [name for name, spec in POLICIES.items() if not spec.needs_segmentation_probe]


@dataclass(frozen=True)
class Matrix:
    """Builds each group's jobs; their outputs share one timestamp."""

    out_dir: Path
    timestamp: str
    num_runs: int
    num_glimpses: int
    include_extra_grids: bool

    def jobs(self, group: Group) -> list[EvaluationJob]:
        match group:
            case "ade20k_seg":
                return self.ade20k_seg()
            case "ade20k_seg_ablations":
                return self.ade20k_seg_ablations()
            case "ade20k_seg_pretrain":
                return self.ade20k_seg_pretrain()
            case "in1k_clf_frozen":
                return self.in1k_clf_frozen()
            case "in1k_clf_finetuned":
                return self.in1k_clf_finetuned()
            case "in1k_clf_ablations":
                return self.in1k_clf_ablations()
            case "in1k_clf_pretrain":
                return self.in1k_clf_pretrain()
            case "recon":
                return self.recon()
            case _:
                assert_never(group)

    def output(self, group: Group, stem: str, run_index: int | None) -> Path:
        run_suffix = "" if run_index is None else f"_r{run_index}"
        return self.out_dir / group / f"{stem}_{self.timestamp}{run_suffix}.pt"

    def canvit_ade20k(
        self, group: Group, *, prefix: str, pretrained_repo: str, probe_model: str, settings: list[Setting],
    ) -> list[EvaluationJob]:
        """Every policy each canvas grid supports, with the probe named after probe_model."""
        jobs = []
        for setting in settings:
            scene, grid = setting.scene_size_px, setting.canvas_grid_size
            probe_repo = repos.released_ade20k_probe(probe_model, scene_size_px=scene, canvas_grid_size=grid)
            for policy in ade20k_policies(grid):
                for run in range(num_runs_of(policy, self.num_runs)):
                    jobs.append(EvaluationJob(group, run, ADE20kSegmentationCanViT(
                        output=self.output(group, f"{prefix}{policy}_s{scene}_c{grid}", run),
                        pretrained_repo=pretrained_repo, probe_repo=probe_repo,
                        episode=EpisodeConfig(policy=policy, num_glimpses=self.num_glimpses, canvas_grid_size=grid),
                        scene_size_px=scene, batch_size=setting.batch_size,
                    )))
        return jobs

    def ade20k_seg(self) -> list[EvaluationJob]:
        """The flagship under every policy, the DINOv3 baselines, and the flagship's first glimpse on each canvas."""
        extra = self.include_extra_grids
        jobs = self.canvit_ade20k(
            "ade20k_seg", prefix="", pretrained_repo=repos.PRETRAINED["in21k"], probe_model="in21k",
            settings=ADE20K_SETTINGS + (EXTRA_ADE20K_SETTINGS if extra else []),
        )
        for variant in DINOV3_REPOS:
            for input_size_px in DINOV3_INPUT_SIZES_PX:
                stem = f"{repos.DINOV3_PROBE_MODEL_NAMES[variant]}_{input_size_px}px"
                jobs.append(EvaluationJob("ade20k_seg", 0, ADE20kSegmentationDINOv3(
                    output=self.output("ade20k_seg", stem, None),
                    variant=variant, input_size_px=input_size_px,
                )))
        for setting in ADE20K_SINGLE_GLIMPSE_SETTINGS + (EXTRA_ADE20K_SINGLE_GLIMPSE_SETTINGS if extra else []):
            scene, grid = setting.scene_size_px, setting.canvas_grid_size
            jobs.append(EvaluationJob("ade20k_seg", 0, ADE20kSegmentationCanViT(
                output=self.output("ade20k_seg", f"canvit_s{scene}_c{grid}", None),
                pretrained_repo=repos.PRETRAINED["in21k"],
                probe_repo=repos.released_ade20k_probe("in21k", scene_size_px=scene, canvas_grid_size=grid),
                episode=EpisodeConfig(policy="coarse_to_fine", num_glimpses=1, canvas_grid_size=grid),
                scene_size_px=scene, batch_size=setting.batch_size,
            )))
        return jobs

    def ade20k_seg_ablations(self) -> list[EvaluationJob]:
        """Every pretraining ablation's checkpoint, with the probe trained on its canvas."""
        return [
            job
            for ablation in ABLATIONS.values()
            for job in self.canvit_ade20k(
                "ade20k_seg_ablations", prefix=f"abl-{ablation.slug}_", pretrained_repo=ablation.released_repo,
                probe_model=ablation.probe_model_name, settings=[ADE20K_SETTINGS[0]],
            )
        ]

    def ade20k_seg_pretrain(self) -> list[EvaluationJob]:
        """Each pretraining dataset's checkpoint, with the probes trained on its canvas."""
        return [
            job
            for dataset, pretrained_repo in repos.PRETRAINED.items()
            for job in self.canvit_ade20k(
                "ade20k_seg_pretrain", prefix=f"pretrain-{dataset}_", pretrained_repo=pretrained_repo,
                probe_model=dataset, settings=ADE20K_PRETRAIN_SETTINGS,
            )
        ]

    def imagenet(
        self, group: Group, *, prefix: str, classifier: FrozenWithFusedProbe | FineTuned,
        policies: list[PolicyName], settings: list[Setting],
    ) -> list[EvaluationJob]:
        jobs = []
        for setting in settings:
            scene, grid = setting.scene_size_px, setting.canvas_grid_size
            for policy in policies:
                for run in range(num_runs_of(policy, self.num_runs)):
                    jobs.append(EvaluationJob(group, run, ImageNetClassification(
                        output=self.output(group, f"{prefix}{policy}_s{scene}_c{grid}", run), classifier=classifier,
                        episode=EpisodeConfig(policy=policy, num_glimpses=self.num_glimpses, canvas_grid_size=grid),
                        scene_size_px=scene, batch_size=setting.batch_size,
                    )))
        return jobs

    def in1k_clf_frozen(self) -> list[EvaluationJob]:
        return self.imagenet(
            "in1k_clf_frozen", prefix="in1k_",
            classifier=FrozenWithFusedProbe(pretrained_repo=repos.PRETRAINED["in21k"]), policies=in1k_policies(),
            settings=IN1K_SETTINGS + (EXTRA_IN1K_SETTINGS if self.include_extra_grids else []),
        )

    def in1k_clf_finetuned(self) -> list[EvaluationJob]:
        """At the scene and canvas sizes of fine-tuning only."""
        return self.imagenet(
            "in1k_clf_finetuned", prefix="in1k_", classifier=FineTuned(), policies=in1k_policies(),
            settings=IN1K_SETTINGS,
        )

    def in1k_clf_ablations(self) -> list[EvaluationJob]:
        """Every pretraining ablation's checkpoint, frozen, under C2F; all fuse the same teacher probe."""
        return [
            job
            for ablation in ABLATIONS.values()
            for job in self.imagenet(
                "in1k_clf_ablations", prefix=f"abl-{ablation.slug}_",
                classifier=FrozenWithFusedProbe(pretrained_repo=ablation.released_repo),
                policies=["coarse_to_fine"], settings=IN1K_SETTINGS,
            )
        ]

    def in1k_clf_pretrain(self) -> list[EvaluationJob]:
        """Each pretraining dataset's checkpoint, frozen."""
        return [
            job
            for dataset, pretrained_repo in repos.PRETRAINED.items()
            for job in self.imagenet(
                "in1k_clf_pretrain", prefix=f"pretrain-{dataset}_",
                classifier=FrozenWithFusedProbe(pretrained_repo=pretrained_repo),
                policies=in1k_policies(), settings=IN1K_SETTINGS,
            )
        ]

    def recon(self) -> list[EvaluationJob]:
        """Every pretraining ablation's checkpoint, under the ablation study's episodes."""
        return [
            EvaluationJob("recon", run, Reconstruction(
                output=self.output("recon", f"recon_{ablation.slug}", run), pretrained_repo=ablation.released_repo,
                episode=ABLATION_EPISODE,
            ))
            for ablation in ABLATIONS.values()
            for run in range(num_runs_of(ABLATION_EPISODE.policy, self.num_runs))
        ]


@dataclass(frozen=True, kw_only=True)
class Batch:
    """The paper's evaluation matrix, one job at a time, each in a fresh process."""

    out_dir: Path = Path("results")
    groups: tuple[Group, ...] = DEFAULT_GROUPS
    num_runs: int = 5
    """Runs of each stochastic policy; deterministic policies run once."""
    num_glimpses: int = NUM_GLIMPSES
    """Glimpses per scene for the ADE20K and ImageNet-1k jobs; reconstruction uses the ablation study's."""
    include_extra_grids: bool = False
    """Add the canvas grid sweep: ADE20K grids 8 to 24 at 512 px, frozen ImageNet-1k grids 8, 16 and 64."""
    device: str = "cuda"
    max_batch_size: int | None = None
    """Cap every job's batch size, for GPUs with less memory."""
    policies: tuple[PolicyName, ...] = ()
    """Keep only the jobs with these policies; empty keeps all."""
    grids: tuple[int, ...] = ()
    """Keep only the jobs with these canvas grids, or DINOv3 patch grids; empty keeps all."""
    shard_index: int = 0
    shard_count: int = 1
    """Run jobs[shard_index::shard_count] of the ordered, filtered list, e.g. one shard per SLURM array task."""
    skip_existing: bool = False
    """Skip jobs whose output exists under any timestamp; applied after sharding, so shards stay stable."""
    dry_run: bool = False
    """Print the jobs instead of running them."""

    def __post_init__(self) -> None:
        assert 0 <= self.shard_index < self.shard_count, (self.shard_index, self.shard_count)
        assert self.max_batch_size is None or self.max_batch_size >= 1, self.max_batch_size

    def jobs(self) -> list[EvaluationJob]:
        """This shard of the ordered, filtered matrix."""
        matrix = Matrix(
            out_dir=self.out_dir, timestamp=time.strftime(TIMESTAMP_FORMAT, time.gmtime()), num_runs=self.num_runs,
            num_glimpses=self.num_glimpses, include_extra_grids=self.include_extra_grids,
        )
        jobs = [
            dataclasses.replace(job, task=self.on_this_machine(job.task))
            for group in self.groups for job in matrix.jobs(group)
        ]
        if self.policies:
            jobs = [job for job in jobs if job.policy in self.policies]
        if self.grids:
            jobs = [job for job in jobs if job.output_grid_size in self.grids]
        jobs.sort(key=lambda job: (job.run_index, job.group, job.output.name))
        return jobs[self.shard_index :: self.shard_count]

    def on_this_machine(self, task: EvaluationTask) -> EvaluationTask:
        """task on this batch's device, with its batch size capped at max_batch_size."""
        cap = self.max_batch_size
        batch_size = task.batch_size if cap is None else min(task.batch_size, cap)
        return dataclasses.replace(task, device=self.device, batch_size=batch_size)

    def run(self) -> None:
        jobs = self.jobs()
        if self.skip_existing:
            jobs = [job for job in jobs if not job.already_done()]
        log.info("%d jobs: %s", len(jobs), dict(Counter(job.group for job in jobs)))
        if self.dry_run:
            for job in jobs:
                print(f"{job.output}  {job.task!r}")
            return
        failed = []
        for i, job in enumerate(jobs, start=1):
            log.info("[%d/%d] %s", i, len(jobs), job.output)
            if not (run_in_fresh_process(job.task.run) and job.output.exists()):
                log.error("[%d/%d] failed: %s", i, len(jobs), job.output)
                failed.append(str(job.output))
        if failed:
            raise SystemExit(f"{len(failed)} of {len(jobs)} jobs failed: {failed}")
        log.info("All %d jobs done", len(jobs))
