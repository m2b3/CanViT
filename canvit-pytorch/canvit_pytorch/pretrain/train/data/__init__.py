"""Data loading for CanViT pretraining: shard loaders + batch types."""

import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from torch import Tensor

if TYPE_CHECKING:
    from ..config import Config
from canvit_pytorch.preprocess import preprocess
from torch.utils.data import DataLoader, Dataset

from canvit_pretrain.datasets import IndexedImageFolder

from .shards import ShardedFeatureLoader

log = logging.getLogger(__name__)

type Batch = tuple[Tensor, ...]  # Generic batch (images, labels, ...)


# IN21k contains corrupt images that cause DataLoader workers to fail.
# Observed PIL errors: "Corrupt EXIF data", "Truncated File Read", UnidentifiedImageError.
# See bad_images.txt for the full list. Skip failed batches up to this limit.
MAX_CONSECUTIVE_FAILURES = 10


class InfiniteLoader:
    """Infinite iterator over a DataLoader with retry on worker errors.

    Note: We use explicit iterator management instead of a generator because
    when an exception propagates out of a Python generator, the generator is
    finalized (gi_frame=None) and subsequent next() calls raise StopIteration.

    Used only for the val loader (map-style IndexedImageFolder dataset).
    """

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._iter: Iterator[Batch] | None = None

    def _next_with_retry(self) -> Batch:
        failures = 0
        while True:
            if self._iter is None:
                self._iter = iter(self._loader)
            try:
                return next(self._iter)
            except StopIteration:
                # End of epoch - start new one
                self._iter = iter(self._loader)
            except Exception as e:
                failures += 1
                log.warning(f"Batch failed ({failures}/{MAX_CONSECUTIVE_FAILURES}): {e}")
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(f"{MAX_CONSECUTIVE_FAILURES} consecutive batch failures") from e
                # Worker error corrupts iterator state - reset it
                self._iter = None

    def next(self) -> Batch:
        """Get next batch (raw tuple from DataLoader)."""
        return self._next_with_retry()

    def next_batch(self) -> Tensor:
        """Get images only (first element of batch)."""
        images, *_ = self._next_with_retry()
        return images

    def next_batch_with_labels(self) -> tuple[Tensor, Tensor]:
        """Get (images, labels) - for raw image loaders."""
        batch = self._next_with_retry()
        return batch[0], batch[1]


class Loaders(NamedTuple):
    """Train and validation data loaders."""

    train: ShardedFeatureLoader
    val: InfiniteLoader


def scene_size_px(grid_size: int, patch_size: int) -> int:
    return grid_size * patch_size


def create_loaders(cfg: "Config", start_step: int) -> Loaders:
    """Train + val loaders; train reads precomputed features, val reads raw images.

    start_step positions the shard cursor on resume (0 for fresh).
    """
    from ..config import Config
    assert isinstance(cfg, Config)
    log.info(f"=== CREATE_LOADERS: start_step={start_step} ===")

    val_dir = cfg.val_dir
    assert val_dir.is_dir(), f"val_dir not found: {val_dir}"

    sz = cfg.scene_resolution
    persistent = cfg.num_workers > 0

    # Train loader: precomputed features (required)
    assert cfg.feature_base_dir is not None, "feature_base_dir required"
    assert (cfg.feature_image_root is None) != (cfg.tar_dir is None), \
        "Exactly one of feature_image_root or tar_dir required"
    log.info("Train: using PRECOMPUTED FEATURES (ShardedFeatureLoader)")
    shards_dir = cfg.feature_base_dir / cfg.teacher_name / str(sz) / "shards"
    log.info(f"  feature_base_dir: {cfg.feature_base_dir}")
    log.info(f"  teacher_name: {cfg.teacher_name}")
    log.info(f"  resolution: {sz}")
    log.info(f"  → shards_dir: {shards_dir}")
    if cfg.tar_dir is not None:
        log.info(f"  tar_dir: {cfg.tar_dir} (images read directly from tar)")
    else:
        log.info(f"  image_root: {cfg.feature_image_root}")
    assert shards_dir.is_dir(), f"shards_dir not found: {shards_dir}"
    train_loader = ShardedFeatureLoader(
        shards_dir=shards_dir,
        image_size=sz,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        start_step=start_step,
        image_root=cfg.feature_image_root,
        tar_dir=cfg.tar_dir,
        steps_per_job=cfg.steps_per_job,
    )
    log.info(f"  {len(train_loader.shard_files)} shards, start_shard={train_loader.start_shard}")

    # Val loader
    val_tf = preprocess(sz)
    if cfg.val_index_dir is not None:
        val_index_dir = cfg.val_index_dir
        log.info(f"Val: using provided val_index_dir={val_index_dir}")
    elif cfg.train_index_dir is not None:
        val_index_dir = cfg.train_index_dir
        log.info(f"Val: val_index_dir not set, using train_index_dir={val_index_dir}")
    else:
        val_index_dir = Path(tempfile.mkdtemp(prefix="avp_val_index_"))
        log.info(f"Val: no index_dir available, using temp dir: {val_index_dir}")
    val_ds: Dataset[tuple] = IndexedImageFolder(val_dir, val_index_dir, val_tf)
    assert len(val_ds) > 0, "val dataset empty"
    log.info(f"Val dataset: {len(val_ds):,} images, resolution: {sz}px")
    # CRITICAL: shuffle=True required! Without it, batches are sequential
    # (all tench, then all goldfish, etc.) which gives misleading metrics.
    val_loader = InfiniteLoader(DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True, persistent_workers=persistent,
    ))

    return Loaders(train=train_loader, val=val_loader)
