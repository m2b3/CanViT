"""Two pretraining jobs on synthetic shards, the second resuming the first: data, steps, checkpoints, monitoring."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from canvit_pytorch.hub.publish import Pretrained
from canvit_pytorch.model.config import CanViTConfig
from canvit_pytorch.pretrain import checkpoint
from canvit_pytorch.pretrain.config import PretrainingConfig
from canvit_pytorch.pretrain.features.shard import STORAGE_DTYPE, FeatureShard, shard_path
from canvit_pytorch.pretrain.loop import train
from canvit_pytorch.teacher import TEACHER_REPO

CANVAS_GRID = 4
SCENE_PX = CANVAS_GRID * 16
TEACHER_DIM = 768
SCENES_PER_SHARD = 8


def _write_image(path: Path, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, (SCENE_PX, SCENE_PX, 3), dtype=np.uint8)).save(path)


def _write_shards(root: Path, rng: np.random.Generator) -> tuple[Path, Path]:
    images_dir, shards_dir = root / "images", root / "shards"
    shards_dir.mkdir(parents=True)
    for shard_id in range(2):
        paths = [f"class{i % 2}/{shard_id}_{i}.png" for i in range(SCENES_PER_SHARD)]
        for p in paths:
            _write_image(images_dir / p, rng)
        start = shard_id * SCENES_PER_SHARD
        shard: FeatureShard = {
            "patches": torch.randn(SCENES_PER_SHARD, CANVAS_GRID**2, TEACHER_DIM).to(STORAGE_DTYPE),
            "cls": torch.randn(SCENES_PER_SHARD, TEACHER_DIM).to(STORAGE_DTYPE),
            "paths": paths, "class_idxs": torch.zeros(SCENES_PER_SHARD, dtype=torch.int32),
            "image_hashes": [""] * SCENES_PER_SHARD, "failed_indices": [],
            "shard_id": shard_id, "start_idx": start, "end_idx": start + SCENES_PER_SHARD,
            "parquet_path": "synthetic", "parquet_sha256": "0" * 16, "teacher_repo_id": TEACHER_REPO,
            "image_size": SCENE_PX, "shard_size": SCENES_PER_SHARD, "dtype": str(STORAGE_DTYPE),
            "embed_dim": TEACHER_DIM, "n_patches": CANVAS_GRID**2, "batch_size": 2,
            "created_at": "", "git_commit": None,
        }
        torch.save(shard, shard_path(shards_dir, shard_id))
    return images_dir, shards_dir


@pytest.mark.slow
@pytest.mark.network
def test_two_jobs_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMET_START_ONLINE", "false")
    monkeypatch.setenv("COMET_OFFLINE_DIRECTORY", str(tmp_path / "comet"))
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    rng = np.random.default_rng(0)
    images_dir, shards_dir = _write_shards(tmp_path, rng)
    validation_dir = tmp_path / "validation"
    for class_idx in range(1000):  # ImageNet-1k's class count is checked
        _write_image(validation_dir / f"n{class_idx:08d}" / "0.png", rng)

    config = PretrainingConfig(
        shards_dir=shards_dir, images_dir=images_dir, validation_dir=validation_dir, dataset="in1k",
        checkpoints_dir=tmp_path / "checkpoints", run_name="smoke",
        model=CanViTConfig(backbone_name="vits16"), glimpse_size_px=32, canvas_grid_size=CANVAS_GRID,
        batch_size=2, warmup_steps=2, total_steps=5, steps_per_job=3,
        device="cpu", compile=False, amp=False, num_workers=0,
        log_every=1, validate_every=2, figure_every=2, validation_glimpses=3,
    )
    train(config)
    first = checkpoint.load(checkpoint.latest(config.checkpoints_dir / "smoke") or Path(), device=torch.device("cpu"))
    assert first["step"] == 3 and len(first["jobs"]) == 1

    train(config)
    second = checkpoint.load(checkpoint.latest(config.checkpoints_dir / "smoke") or Path(), device=torch.device("cpu"))
    assert second["step"] == 5 and len(second["jobs"]) == 2
    assert second["comet_experiment_key"] == first["comet_experiment_key"]
    assert not torch.equal(second["model"]["canvit.init_canvas_patch"], first["model"]["canvit.init_canvas_patch"])
    model = checkpoint.model_from_checkpoint(second)
    assert bool(model.teacher_patch_standardizer.fitted) and bool(model.teacher_cls_standardizer.fitted)

    train(config)  # the run is complete: no new checkpoint
    final = config.checkpoints_dir / "smoke" / "step-5.pt"
    assert checkpoint.latest(config.checkpoints_dir / "smoke") == final
    assert not (config.checkpoints_dir / "smoke" / "FAILED").exists()
    assert os.listdir(tmp_path / "comet")

    Pretrained(checkpoint=final, repo="canvit/smoke", out_dir=tmp_path / "staging", ablation="no-reads").run()
    card = (tmp_path / "staging" / "smoke" / "README.md").read_text()
    assert "No canvas reads" in card and "| Training steps | 5 |" in card
