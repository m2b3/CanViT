"""The probe-training loop on a tiny synthetic ADE20K tree, with a deterministic stand-in for the frozen model."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from canvit_pytorch import SegmentationProbe
from canvit_pytorch.benchmarks.ade20k.dataset import Split
from canvit_pytorch.specialize.ade20k import data
from canvit_pytorch.specialize.ade20k.config import ProbeTrainingConfig
from canvit_pytorch.specialize.ade20k.record import RECORD_FILENAME
from canvit_pytorch.specialize.ade20k.train import feature_autocast, train_probe, validate

SCENE_SIZE_PX = 64
POOLING_PX = 8


@dataclass(frozen=True)
class PooledPixels:
    """Two feature maps: the scene average-pooled in POOLING_PX cells, and its negation."""

    layer_normalized: ClassVar[bool] = False
    embed_dim: int = 3

    def __call__(self, images: Tensor, split: Split) -> list[Tensor]:
        pooled = F.avg_pool2d(images, POOLING_PX).permute(0, 2, 3, 1)
        return [pooled, -pooled]


@dataclass(frozen=True)
class PooledPixelsProbeConfig:
    training: ProbeTrainingConfig
    run_name: str = "pooled-pixels-probe"

    def load_feature_maps(self, device: torch.device) -> PooledPixels:
        return PooledPixels()


def write_fake_ade20k(root: Path) -> None:
    rng = np.random.default_rng(0)
    for split, count in (("training", 4), ("validation", 3)):
        (root / "images" / split).mkdir(parents=True)
        (root / "annotations" / split).mkdir(parents=True)
        for i in range(count):
            labels = np.kron(rng.integers(0, 151, (6, 8), dtype=np.uint8), np.ones((8, 8), dtype=np.uint8))
            image = (rng.random((*labels.shape, 3)) * 64 + labels[..., None]).astype(np.uint8)
            Image.fromarray(image).save(root / "images" / split / f"{i}.jpg")
            Image.fromarray(labels, mode="L").save(root / "annotations" / split / f"{i}.png")


def test_the_recorded_probe_reproduces_its_recorded_validation_miou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_fake_ade20k(tmp_path / "ade20k")
    monkeypatch.setenv("ADE20K_ROOT", str(tmp_path / "ade20k"))
    monkeypatch.setenv("COMET_AUTO_LOG_DISABLE", "1")
    training = ProbeTrainingConfig(
        output_dir=tmp_path / "runs", device="cpu", scene_size_px=SCENE_SIZE_PX, num_steps=4, batch_size=2,
        warmup_steps=1, feature_dtype="float32", validation_batch_size=2, num_workers=0, validate_every=2,
        log_every=1, figure_every=2, figure_samples=2,
    )

    run_dir = train_probe(PooledPixelsProbeConfig(training=training))

    record = json.loads((run_dir / RECORD_FILENAME).read_text())
    assert record["steps_trained"] == training.num_steps
    assert [path.name for path in run_dir.glob("probe_step*")] == [record["probe_dir"]]
    probe = SegmentationProbe.from_pretrained(str(run_dir / record["probe_dir"]))
    device = torch.device("cpu")
    validation_miou, _ = validate(
        probe=probe, feature_maps=PooledPixels(), loader=data.validation_loader(training),
        autocast=feature_autocast(training, device), device=device, figure_samples=training.figure_samples,
    )
    assert validation_miou == record["validation_miou"]
