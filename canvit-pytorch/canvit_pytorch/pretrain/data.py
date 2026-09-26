"""Batches for pretraining: scenes with their precomputed teacher features, and validation images.

Training reads the shards in order, forever; DataLoader workers split each
shard's images. Step N resumes at the batch it would have read, so a run split
across jobs sees the same data as an uninterrupted one.
"""

import itertools
import logging
from collections.abc import Iterator
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from canvit_pytorch.benchmarks.imagenet import validation_set
from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.pretrain.features.shard import load_shard

log = logging.getLogger(__name__)


class ShardSamples(IterableDataset[tuple[Tensor, Tensor, Tensor]]):
    """(image, teacher patch features, teacher CLS) of every shard in turn, from start_shard on."""

    def __init__(self, *, shard_paths: list[Path], images_dir: Path, scene_size_px: int, start_shard: int) -> None:
        self.shard_paths = shard_paths
        self.images_dir = images_dir
        self.scene_size_px = scene_size_px
        self.start_shard = start_shard

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker is not None else (0, 1)
        transform = preprocess(self.scene_size_px)
        for shard_index in itertools.count(self.start_shard):
            path = self.shard_paths[shard_index % len(self.shard_paths)]
            shard = load_shard(path)
            failed = set(shard["failed_indices"])
            if worker_id == 0:
                log.info(f"Reading shard {shard_index} ({path.name}; {len(failed)} failed images skipped)")
            for i in range(worker_id, len(shard["paths"]), num_workers):
                if i in failed:
                    continue
                image_path = self.images_dir / shard["paths"][i]
                try:
                    with Image.open(image_path) as file:
                        image = file.convert("RGB")
                except Exception as e:  # noqa: BLE001  ImageNet-21k holds corrupt images that the export did not flag
                    log.warning(f"Skipping unreadable {image_path}: {e}")
                    continue
                scene = transform(image)
                assert isinstance(scene, Tensor)
                yield scene, shard["patches"][i].clone(), shard["cls"][i].clone()


class TrainingBatches:
    """(images, teacher patch features, teacher CLS) batches from the shards, starting at optimizer step start_step."""

    def __init__(
        self, *, shards_dir: Path, images_dir: Path, scene_size_px: int, teacher_repo: str,
        batch_size: int, num_workers: int, start_step: int,
    ) -> None:
        shard_paths = sorted(shards_dir.glob("*.pt"))
        assert shard_paths, f"no shards in {shards_dir}"
        first = load_shard(shard_paths[0])
        assert (first["teacher_repo_id"], first["image_size"]) == (teacher_repo, scene_size_px), (
            f"{shards_dir} holds {first['teacher_repo_id']} features of {first['image_size']} px scenes; "
            f"this run needs {teacher_repo} at {scene_size_px} px"
        )
        batches_per_shard = len(first["paths"]) // batch_size
        start_shard, self._batches_to_skip = divmod(start_step, batches_per_shard)
        log.info(f"{len(shard_paths)} shards of {batches_per_shard} batches; step {start_step} starts in shard {start_shard}")
        self.loader = DataLoader(
            ShardSamples(shard_paths=shard_paths, images_dir=images_dir, scene_size_px=scene_size_px, start_shard=start_shard),
            batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True,
            persistent_workers=num_workers > 0,
        )
        self._iterator: Iterator[list[Tensor]] | None = None

    def next(self) -> tuple[Tensor, Tensor, Tensor]:
        if self._iterator is None:
            self._iterator = iter(self.loader)
            for _ in range(self._batches_to_skip):
                next(self._iterator)
        images, patches, cls = next(self._iterator)
        return images, patches, cls


class ValidationBatches:
    """Shuffled (images, labels) batches of ImageNet-1k validation images, forever."""

    def __init__(self, *, validation_dir: Path, scene_size_px: int, batch_size: int, num_workers: int) -> None:
        self.loader = DataLoader(
            validation_set(validation_dir, size_px=scene_size_px), batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
        )
        self._iterator: Iterator[list[Tensor]] | None = None

    def next(self) -> tuple[Tensor, Tensor]:
        if self._iterator is None:
            self._iterator = iter(self.loader)
        try:
            images, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            images, labels = next(self._iterator)
        except Exception:
            self._iterator = None  # a failed worker leaves the iterator unusable
            raise
        return images, labels
