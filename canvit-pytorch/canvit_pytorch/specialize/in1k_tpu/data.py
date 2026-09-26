"""ImageNet-1k from TFRecord shards, in batches of glimpses that the data-loader workers take along a viewing policy."""
# pyright: reportMissingImports=false
# (tfrecord is installed only in the tpu/ environment.)

import io
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, assert_never

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from tfrecord.reader import tfrecord_loader
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from canvit_pytorch.hub.repos import RELEASED_SCENE_SIZE_PX
from canvit_pytorch.policies import FixedSequence, PolicyName, make_policy
from canvit_pytorch.preprocess import IMAGENET_MEAN, IMAGENET_STD
from canvit_pytorch.viewpoint import sample_at_viewpoint

log = logging.getLogger(__name__)

SCENE_SIZE_PX = RELEASED_SCENE_SIZE_PX
TFRECORD_FIELDS = {"image/encoded": "byte", "image/class/label": "int"}

ShardSplit = Literal["train", "validation"]
"""The file-name prefix of a split's shards."""

TRAINING_TRANSFORM = T.Compose([
    T.RandomResizedCrop(SCENE_SIZE_PX, scale=(0.2, 1.0)),  # bilinear, as in pretraining
    T.RandomHorizontalFlip(),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

VALIDATION_TRANSFORM = T.Compose([
    T.Resize(SCENE_SIZE_PX),  # bilinear, as in pretraining
    T.CenterCrop(SCENE_SIZE_PX),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

Transform = Callable[[Image.Image], Tensor]


class GlimpseBatch(NamedTuple):
    glimpses: Tensor  # [N, B, 3, g, g]: glimpse t of every scene
    labels: Tensor  # [B] class indices
    centers: Tensor  # [N, B, 2] viewpoint centers
    scales: Tensor  # [N, B] viewpoint scales


def decode_record(record: dict[str, Any], transform: Transform) -> tuple[Tensor, int]:
    """(scene, class index) of a TFRecord, whose labels count classes from 1."""
    label = int(record["image/class/label"].item()) - 1
    # XXX: a record labeled 0 would silently become class 0. ImageNet TFRecords label classes 1 to 1000, so an
    # assertion could replace this clamp, which the published checkpoint's run had.
    label = max(0, label)
    image = Image.open(io.BytesIO(record["image/encoded"])).convert("RGB")
    return transform(image), label


class TFRecordShards(IterableDataset[tuple[Tensor, int]]):
    """Endless passes over the shards; each data-loader worker reads its own shards, in a new random order each pass.

    A record that fails to decode is skipped with a warning that counts the worker's skips.
    """

    def __init__(self, shards: list[Path], transform: Transform) -> None:
        assert shards, "no shards"
        self.shards = shards
        self.transform = transform

    def __iter__(self) -> Iterator[tuple[Tensor, int]]:
        worker = get_worker_info()
        shards = self.shards if worker is None else self.shards[worker.id :: worker.num_workers]
        rng = np.random.default_rng()
        num_skipped = 0
        while True:
            for shard_index in rng.permutation(len(shards)):
                shard = shards[shard_index]
                for record_index, record in enumerate(tfrecord_loader(str(shard), None, TFRECORD_FIELDS)):
                    try:
                        sample = decode_record(record, self.transform)
                    except Exception as error:  # noqa: BLE001  a corrupt record is skipped, logged and counted
                        num_skipped += 1
                        log.warning("Skipped record %d of %s (%d skipped by this worker): %r",
                                    record_index, shard, num_skipped, error)
                        continue
                    yield sample


def find_shards(data_dir: Path, split: ShardSplit) -> list[Path]:
    shards = sorted(path for path in data_dir.glob(f"{split}-*") if not path.suffix)
    assert shards, f"no {split} shards in {data_dir}"
    log.info("Found %d %s shards in %s", len(shards), split, data_dir)
    return shards


@dataclass(frozen=True)
class GlimpseCollate:
    """Batches scenes and takes their glimpses at viewpoints the policy draws for the batch."""

    policy: PolicyName
    num_glimpses: int
    glimpse_size_px: int
    canvas_grid_size: int

    def __call__(self, samples: list[tuple[Tensor, int]]) -> GlimpseBatch:
        scenes = torch.stack([scene for scene, _ in samples])
        labels = torch.tensor([label for _, label in samples], dtype=torch.long)
        policy = make_policy(
            self.policy, batch_size=len(samples), device=scenes.device, num_glimpses=self.num_glimpses,
            canvas_grid_size=self.canvas_grid_size,
        )
        assert isinstance(policy, FixedSequence), f"{self.policy} chooses viewpoints during the episode, not in advance"
        viewpoints = policy.viewpoints
        return GlimpseBatch(
            glimpses=torch.stack([
                sample_at_viewpoint(spatial=scenes, viewpoint=viewpoint, glimpse_size_px=self.glimpse_size_px)
                for viewpoint in viewpoints
            ]),
            labels=labels,
            centers=torch.stack([viewpoint.centers for viewpoint in viewpoints]),
            scales=torch.stack([viewpoint.scales for viewpoint in viewpoints]),
        )


def glimpse_loader(
    *, data_dir: Path, split: ShardSplit, collate: GlimpseCollate, batch_size: int, num_workers: int,
) -> DataLoader[tuple[Tensor, int]]:
    match split:
        case "train":
            transform = TRAINING_TRANSFORM
        case "validation":
            transform = VALIDATION_TRANSFORM
        case _:
            assert_never(split)
    return DataLoader(
        TFRecordShards(find_shards(data_dir, split), transform),
        batch_size=batch_size, num_workers=num_workers, drop_last=True, collate_fn=collate,
        prefetch_factor=2 if num_workers > 0 else None,
    )
