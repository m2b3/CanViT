"""Shard-based feature loading with deterministic resume.

Design:
- Single IterableDataset that iterates over ALL shards sequentially
- Workers split SAMPLES within each shard (not shards across workers)
- Deterministic order: shard 0, 1, 2, ..., n-1, 0, 1, ... (no shuffling)
- Resume via start_shard = start_step // batches_per_shard
- Persistent workers work naturally (dataset controls iteration)

Image sources:
- image_root: images on filesystem, loaded via PIL
- tar_dir: images read directly from mmap'd tar files, no extraction
"""

import logging
import math
import time
from collections.abc import Iterator
from pathlib import Path

import torch
from canvit_pytorch.preprocess import preprocess
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .tar_images import TarImageReader, TarIndex, load_tar_index, scan_tar_headers

log = logging.getLogger(__name__)


class AllShardsDataset(IterableDataset[tuple[Tensor, Tensor, Tensor, int]]):
    """IterableDataset that iterates over all shards sequentially, forever.

    Workers split samples within each shard via `sample_idx % num_workers == worker_id`.
    Resume is handled via `start_shard` parameter.
    """

    def __init__(
        self,
        shard_files: list[Path],
        image_size: int,
        start_shard: int = 0,
        expected_samples_per_shard: int | None = None,
        *,
        image_root: Path | None = None,
        tar_dir: Path | None = None,
        tar_indexes: dict[str, TarIndex] | None = None,
    ) -> None:
        assert (image_root is None) != (tar_dir is None), \
            "Exactly one of image_root or tar_dir must be set"
        self.shard_files = shard_files
        self.image_root = Path(image_root) if image_root is not None else None
        self.tar_dir = Path(tar_dir) if tar_dir is not None else None
        self.image_size = image_size
        self.start_shard = start_shard
        self.expected_samples_per_shard = expected_samples_per_shard
        self.tar_indexes = tar_indexes or {}
        self.transform = preprocess(image_size)

    def _tar_path_for_shard(self, shard_path: Path) -> Path:
        assert self.tar_dir is not None
        return self.tar_dir / f"{shard_path.stem}.tar"

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor, int]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        n_shards = len(self.shard_files)
        shard_counter = self.start_shard
        tar_reader: TarImageReader | None = None

        while True:
            shard_idx = shard_counter % n_shards
            shard_path = self.shard_files[shard_idx]

            # All workers load same shard (mmap=cheap), but each processes different samples:
            # worker i gets samples i, i+num_workers, i+2*num_workers, ...
            t0 = time.perf_counter()
            shard = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
            t_load = time.perf_counter() - t0

            # Open tar for this shard (mmap'd, shared across workers via fork COW)
            if self.tar_dir is not None:
                if tar_reader is not None:
                    tar_reader.close()
                tar_path = self._tar_path_for_shard(shard_path)
                pre_index = self.tar_indexes.get(shard_path.stem)
                # Training path: pre-built index from main process (via fork COW).
                # Bench/test path: no pre-built index, scan headers on the fly.
                index = pre_index if pre_index is not None else scan_tar_headers(tar_path)
                tar_reader = TarImageReader(tar_path, index=index)

            n_samples = len(shard["paths"])
            failed_indices = set(shard.get("failed_indices", []))

            if self.expected_samples_per_shard is not None and n_samples != self.expected_samples_per_shard:
                log.warning(
                    f"Shard {shard_path.name}: {n_samples} samples, expected {self.expected_samples_per_shard}. "
                    f"Resume calculation may be off."
                )

            if worker_id == 0:
                log.info(
                    f"Shard {shard_idx}/{n_shards} ({shard_path.name}): "
                    f"{n_samples} samples, loaded in {t_load:.3f}s"
                )
                if failed_indices:
                    log.warning(f"  {len(failed_indices)} pre-marked failures")

            yielded = 0
            skipped = 0
            t_read = t_transform = t_patches = 0.0
            max_read = max_patches = 0.0
            n_slow_patches = 0  # patches clone > 500ms

            for i in range(worker_id, n_samples, num_workers):
                if i in failed_indices:
                    skipped += 1
                    continue

                rel_path = shard["paths"][i]
                try:
                    t0 = time.perf_counter()
                    if tar_reader is not None:
                        img = tar_reader.read_image(rel_path)
                    else:
                        assert self.image_root is not None
                        with Image.open(self.image_root / rel_path) as f:
                            img = f.convert("RGB")
                    t1 = time.perf_counter()
                    img_tensor = self.transform(img)
                    t2 = time.perf_counter()
                except Exception as e:
                    log.warning(f"Worker {worker_id}: RUNTIME FAILURE {rel_path}: {e}")
                    skipped += 1
                    continue

                dt_read = t1 - t0
                t_read += dt_read
                t_transform += t2 - t1
                max_read = max(max_read, dt_read)

                assert isinstance(img_tensor, Tensor)
                t0 = time.perf_counter()
                patches = shard["patches"][i].clone()
                cls = shard["cls"][i].clone()
                dt_patches = time.perf_counter() - t0
                t_patches += dt_patches
                max_patches = max(max_patches, dt_patches)
                if dt_patches > 0.5:
                    n_slow_patches += 1

                yield (img_tensor, patches, cls, int(shard["class_idxs"][i]))
                yielded += 1

            # All workers log — see per-worker distribution across shard
            total = t_read + t_transform + t_patches
            if yielded > 0 and total > 0:
                log.info(
                    f"w{worker_id} shard {shard_idx}: {yielded} imgs, "
                    f"read={t_read/yielded*1e3:.1f}ms(max={max_read*1e3:.0f}) "
                    f"transform={t_transform/yielded*1e3:.1f}ms "
                    f"patches={t_patches/yielded*1e3:.1f}ms(max={max_patches*1e3:.0f}) "
                    f"slow_patches={n_slow_patches} "
                    f"total={total/yielded*1e3:.1f}ms/img ({yielded/total:.0f} img/s)"
                )

            shard_counter += 1


class ShardedFeatureLoader:
    """Infinite loader over shards with checkpoint/resume support.

    Resume is exact: jumps to the right shard, then skips partial-shard batches.
    """

    def __init__(
        self,
        shards_dir: Path,
        image_size: int,
        batch_size: int,
        num_workers: int,
        start_step: int,
        *,
        image_root: Path | None = None,
        tar_dir: Path | None = None,
        steps_per_job: int | None = None,
    ) -> None:
        assert (image_root is None) != (tar_dir is None), \
            "Exactly one of image_root or tar_dir must be set"
        self.shards_dir = Path(shards_dir)
        self.image_root = image_root
        self.tar_dir = tar_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        log.info(f"Globbing shards in {self.shards_dir}...")
        t0 = time.perf_counter()
        all_shards = sorted(self.shards_dir.glob("*.pt"))
        log.info(f"  Found {len(all_shards)} shards in {time.perf_counter() - t0:.2f}s")
        assert all_shards, f"No shards found in {shards_dir}"

        # When using tar-based loading, only keep shards that have a .tar.idx file.
        # Exports may land new shards after the last idx build — those are unusable.
        if tar_dir is not None:
            td = Path(tar_dir)
            usable = [s for s in all_shards if (td / f"{s.stem}.tar.idx").exists()]
            if len(usable) < len(all_shards):
                log.warning(f"  Filtered to {len(usable)}/{len(all_shards)} shards (rest lack .tar.idx)")
            self.shard_files = usable
        else:
            self.shard_files = all_shards
        assert self.shard_files, f"No usable shards in {shards_dir}"

        # Read first shard to get samples_per_shard (all shards same size)
        # mmap=True: shards may be tens of GB, don't load into RAM just to count samples
        first_shard = torch.load(self.shard_files[0], map_location="cpu", weights_only=False, mmap=True)
        self.samples_per_shard = len(first_shard["paths"])
        del first_shard
        self.batches_per_shard = self.samples_per_shard // batch_size
        self.start_shard = start_step // self.batches_per_shard
        self._skip_batches = start_step % self.batches_per_shard
        log.info(
            f"  {self.samples_per_shard} samples/shard, {self.batches_per_shard} batches/shard, "
            f"start_shard={self.start_shard}, skip={self._skip_batches}"
        )

        # Load pre-built tar indexes (.idx files) in main process.
        # Workers inherit via fork COW — zero indexing during training.
        # .idx files are created by sa1b/build_tar_indexes.py.
        self.tar_indexes: dict[str, TarIndex] = {}
        if self.tar_dir is not None:
            self._prebuild_tar_indexes(steps_per_job)

        # Will be created lazily on first iteration
        self.loader: DataLoader | None = None
        self.loader_iter: Iterator | None = None

    def _prebuild_tar_indexes(self, steps_per_job: int | None) -> None:
        """Load .idx files for upcoming shards. Runs in main process before fork."""
        assert self.tar_dir is not None
        n_shards = len(self.shard_files)
        if steps_per_job is not None:
            n_ahead = math.ceil(steps_per_job / self.batches_per_shard) + 1
        else:
            n_ahead = n_shards
        n_ahead = min(n_ahead, n_shards)

        log.info(f"Loading tar indexes for {n_ahead}/{n_shards} shards...")
        t0 = time.perf_counter()
        for i in range(n_ahead):
            idx = (self.start_shard + i) % n_shards
            sf = self.shard_files[idx]
            tar_path = self.tar_dir / f"{sf.stem}.tar"
            assert tar_path.exists(), f"Tar not found: {tar_path}"
            self.tar_indexes[sf.stem] = load_tar_index(tar_path)
        elapsed = time.perf_counter() - t0
        log.info(f"Loaded {len(self.tar_indexes)} tar indexes in {elapsed:.1f}s")

    def _create_loader(self) -> DataLoader:
        """Create DataLoader with dataset starting at start_shard."""
        dataset = AllShardsDataset(
            shard_files=self.shard_files,
            image_size=self.image_size,
            start_shard=self.start_shard,
            expected_samples_per_shard=self.samples_per_shard,
            image_root=self.image_root,
            tar_dir=self.tar_dir,
            tar_indexes=self.tar_indexes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def next(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get next batch."""
        if self.loader is None:
            log.info(f"Creating DataLoader with {self.num_workers} workers, persistent={self.num_workers > 0}")
            self.loader = self._create_loader()
            self.loader_iter = iter(self.loader)
            if self._skip_batches > 0:
                log.info(f"Skipping {self._skip_batches} batches for exact resume")
                for _ in range(self._skip_batches):
                    next(self.loader_iter)
                self._skip_batches = 0

        assert self.loader_iter is not None
        return next(self.loader_iter)
