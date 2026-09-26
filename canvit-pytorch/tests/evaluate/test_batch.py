"""The evaluation matrix: output names the paper pipeline parses, and the configurations behind them."""

import re
from pathlib import Path
from typing import Any, get_args

import pytest

from canvit_pytorch.evaluate.batch import IN1K_SETTINGS, Batch, EvaluationJob, Group
from canvit_pytorch.evaluate.tasks.ade20k_segmentation import ADE20kSegmentationCanViT, ADE20kSegmentationDINOv3
from canvit_pytorch.evaluate.tasks.imagenet_classification import (
    FineTuned,
    FrozenWithFusedProbe,
    ImageNetClassification,
)
from canvit_pytorch.evaluate.tasks.reconstruction import Reconstruction
from canvit_pytorch.hub import repos
from canvit_pytorch.policies import POLICIES
from canvit_pytorch.pretrain.ablations import ABLATIONS

ALL_GROUPS: tuple[Group, ...] = get_args(Group)
TIMESTAMP = r"\d{8}T\d{6}Z"
# The file names of each output directory, as the paper's figure and table pipeline parses them.
POLICY = rf"(?P<policy>{'|'.join(POLICIES)})_s(?P<scene>\d+)_c(?P<grid>\d+)"
ABLATION = rf"abl-(?P<slug>{'|'.join(ABLATIONS)})"
PRETRAINED = r"pretrain-(?P<dataset>in21k|in1k)"
RUN = rf"_{TIMESTAMP}_r\d+"
NAME_PATTERNS: dict[Group, list[str]] = {
    "ade20k_seg": [
        rf"{POLICY}{RUN}",
        rf"(?P<variant>dv3[bs])_(?P<input>\d+)px_{TIMESTAMP}",
        rf"canvit_s(?P<scene>\d+)_c(?P<grid>\d+)_{TIMESTAMP}",
    ],
    "ade20k_seg_ablations": [rf"{ABLATION}_{POLICY}{RUN}"],
    "ade20k_seg_pretrain": [rf"{PRETRAINED}_{POLICY}{RUN}"],
    "in1k_clf_frozen": [rf"in1k_{POLICY}{RUN}"],
    "in1k_clf_finetuned": [rf"in1k_{POLICY}{RUN}"],
    "in1k_clf_ablations": [rf"{ABLATION}_{POLICY}{RUN}"],
    "in1k_clf_pretrain": [rf"{PRETRAINED}_{POLICY}{RUN}"],
    "recon": [rf"recon_(?P<slug>{'|'.join(ABLATIONS)}){RUN}"],
}


def all_jobs(tmp_path: Path, **options: Any) -> list[EvaluationJob]:
    return Batch(out_dir=tmp_path, groups=ALL_GROUPS, include_extra_grids=True, **options).jobs()


def expected_models(fields: dict[str, str]) -> tuple[str, str]:
    """(pretrained repo, the model name in its ADE20K probes' names) that a file name designates."""
    if "slug" in fields:
        ablation = next(ablation for ablation in ABLATIONS.values() if ablation.slug == fields["slug"])
        return ablation.released_repo, ablation.probe_model_name
    dataset = fields.get("dataset", "in21k")
    return next(repo for name, repo in repos.PRETRAINED.items() if name == dataset), dataset


def test_output_names_encode_each_jobs_configuration(tmp_path: Path) -> None:
    jobs = all_jobs(tmp_path, num_runs=2)
    assert len({job.output for job in jobs}) == len(jobs), "two jobs would write the same file"
    for job in jobs:
        assert job.output.parent == tmp_path / job.group and job.output.suffix == ".pt"
        matches = [m for pattern in NAME_PATTERNS[job.group] if (m := re.fullmatch(pattern, job.output.stem))]
        assert len(matches) == 1, job.output
        fields = matches[0].groupdict()
        task = job.task
        if not isinstance(task, ADE20kSegmentationDINOv3):
            assert fields.get("policy", task.episode.policy) == task.episode.policy, job.output
            assert int(fields.get("grid", task.episode.canvas_grid_size)) == task.episode.canvas_grid_size, job.output
        pretrained_repo, probe_model = expected_models(fields)
        match task:
            case ADE20kSegmentationCanViT():
                assert (int(fields["scene"]), task.pretrained_repo) == (task.scene_size_px, pretrained_repo), job.output
                assert task.probe_repo == repos.released_ade20k_probe(
                    probe_model, scene_size_px=task.scene_size_px, canvas_grid_size=task.episode.canvas_grid_size,
                ), job.output
                assert task.episode.num_glimpses == (1 if job.output.stem.startswith("canvit_") else 21), job.output
            case ADE20kSegmentationDINOv3():
                assert fields["variant"] == repos.DINOV3_PROBE_MODEL_NAMES[task.variant], job.output
                assert int(fields["input"]) == task.input_size_px, job.output
            case ImageNetClassification():
                assert int(fields["scene"]) == task.scene_size_px, job.output
                if job.group == "in1k_clf_finetuned":
                    fine_tuning_settings = {setting[:2] for setting in IN1K_SETTINGS}
                    assert task.classifier == FineTuned(), job.output
                    assert (task.scene_size_px, task.episode.canvas_grid_size) in fine_tuning_settings, job.output
                else:
                    assert task.classifier == FrozenWithFusedProbe(pretrained_repo=pretrained_repo), job.output
            case Reconstruction():
                assert task.pretrained_repo == pretrained_repo, job.output


def test_every_configuration_runs_before_any_repeats(tmp_path: Path) -> None:
    jobs = all_jobs(tmp_path, num_runs=3)
    runs = [job.run_index for job in jobs]
    assert runs == sorted(runs)
    for job in jobs:
        assert job.policy is None or job.run_index < (1 if POLICIES[job.policy].deterministic else 3)
    assert {job.run_index for job in jobs if job.policy == "coarse_to_fine"} == {0, 1, 2}


def test_shards_partition_the_jobs(tmp_path: Path) -> None:
    def names(shard_index: int, shard_count: int) -> list[str]:
        batch = Batch(
            out_dir=tmp_path, groups=("recon", "in1k_clf_ablations"), shard_index=shard_index, shard_count=shard_count,
        )
        return [re.sub(TIMESTAMP, "", job.output.name) for job in batch.jobs()]

    everything = names(0, 1)
    for count in (1, 7, len(everything) + 3):
        shards = [names(i, count) for i in range(count)]
        assert sorted(name for shard in shards for name in shard) == sorted(everything)
        assert max(map(len, shards)) - min(map(len, shards)) <= 1
    with pytest.raises(AssertionError):
        Batch(shard_index=3, shard_count=3)


def test_filters_device_and_batch_size_cap(tmp_path: Path) -> None:
    by_policy = all_jobs(tmp_path, policies=("coarse_to_fine",))
    assert by_policy and all(job.policy == "coarse_to_fine" for job in by_policy)
    by_grid = all_jobs(tmp_path, grids=(8,))
    assert {job.output_grid_size for job in by_grid} == {8}
    assert any(isinstance(job.task, ADE20kSegmentationDINOv3) for job in by_grid), "DINOv3 at 128 px: an 8² patch grid"
    on_cpu = all_jobs(tmp_path, device="cpu", max_batch_size=4)
    assert {(job.task.device, job.task.batch_size) for job in on_cpu} == {("cpu", 4)}


def test_skip_existing_matches_any_timestamp_but_not_other_runs(tmp_path: Path) -> None:
    earlier = tmp_path / "ade20k_seg" / "coarse_to_fine_s512_c32_20260101T000000Z_r0.pt"
    earlier.parent.mkdir()
    earlier.touch()
    jobs = {job.output.name: job for job in Batch(out_dir=tmp_path, num_runs=2).jobs()}
    done = {name for name, job in jobs.items() if job.already_done()}
    assert len(done) == 1 and re.fullmatch(rf"coarse_to_fine_s512_c32_{TIMESTAMP}_r0\.pt", done.pop())
