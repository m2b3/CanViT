"""Benchmark ShardedFeatureLoader throughput.

Run on cluster:
    source slurm/env.sh
    uv run python scripts/bench_dataloader.py --n-batches 500 --num-workers 16

For interactive exploration:
    uv run ipython -i scripts/bench_dataloader.py -- --n-batches 100
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import tyro
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class Config:
    # Paths - feature_base_dir should include dataset (e.g. $FEATURES_DIR/in21k)
    feature_base_dir: Path = (
        Path(os.environ.get("FEATURES_DIR", "~/canvit-storage/dinov3_dense_features")).expanduser()
        / "in21k"
    )
    image_root: Path = Path(os.environ.get("IN21K_IMAGE_DIR", "/datashare/imagenet/winter21_whole"))
    teacher_model: str = "dinov3_vitb16"
    image_size: int = 512

    # Loader config
    batch_size: int = 64
    num_workers: int = 16

    # Benchmark config
    n_batches: int = 500
    warmup_batches: int = 10


def main(cfg: Config) -> None:
    from canvit_pretrain.train.data import ShardedFeatureLoader

    shards_dir = cfg.feature_base_dir / cfg.teacher_model / str(cfg.image_size) / "shards"
    log.info("=" * 60)
    log.info("ShardedFeatureLoader Benchmark")
    log.info("=" * 60)
    log.info(f"shards_dir: {shards_dir}")
    log.info(f"image_root: {cfg.image_root}")
    log.info(f"batch_size: {cfg.batch_size}")
    log.info(f"num_workers: {cfg.num_workers}")
    log.info(f"n_batches: {cfg.n_batches}")

    assert shards_dir.is_dir(), f"shards_dir not found: {shards_dir}"
    assert cfg.image_root.is_dir(), f"image_root not found: {cfg.image_root}"

    loader = ShardedFeatureLoader(
        shards_dir=shards_dir,
        image_root=cfg.image_root,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        start_step=0,
    )
    log.info(f"Found {len(loader.shard_files)} shards, start_shard={loader.start_shard}")

    # Warmup
    log.info(f"Warming up ({cfg.warmup_batches} batches)...")
    for _ in range(cfg.warmup_batches):
        _ = loader.next()
    log.info("Warmup done")

    # Benchmark
    log.info(f"Benchmarking ({cfg.n_batches} batches)...")
    t0 = time.perf_counter()
    pbar = tqdm(range(cfg.n_batches), desc="Batches", unit="batch")

    for _ in pbar:
        batch = loader.next()
        images, patches, cls, labels = batch
        pbar.set_postfix({"img": f"{images.shape}"})

    elapsed = time.perf_counter() - t0
    batches_per_sec = cfg.n_batches / elapsed
    images_per_sec = cfg.n_batches * cfg.batch_size / elapsed

    log.info("=" * 60)
    log.info("Results")
    log.info("=" * 60)
    log.info(f"Total time: {elapsed:.2f}s")
    log.info(f"Batches/sec: {batches_per_sec:.2f}")
    log.info(f"Images/sec: {images_per_sec:.2f}")

    # Interactive: export for ipython
    globals()["loader"] = loader
    globals()["cfg"] = cfg
    log.info("Exported 'loader' and 'cfg' for interactive use")


if __name__ == "__main__":
    main(tyro.cli(Config))
