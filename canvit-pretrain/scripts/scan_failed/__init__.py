"""Scan feature shards for failed indices, build master index.

Efficiently scans shards in parallel using mmap (tensor data never paged in).
Outputs a parquet with global failed indices for downstream filtering.

Usage:
    uv run python -m scripts.scan_failed --shards-dir /path/to/shards -w 32
"""

import logging
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import torch
import tyro
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()[:12]
    except Exception:
        return "unknown"


def extract_failed(path: Path) -> dict:
    """Extract failed info + paths. mmap avoids loading tensors."""
    data = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    paths = data["paths"]
    failed_indices = data["failed_indices"]
    return {
        "shard_id": data["shard_id"],
        "start_idx": data["start_idx"],
        "shard_size": data["end_idx"] - data["start_idx"],
        "failed_indices": failed_indices,
        "failed_paths": [paths[i] for i in failed_indices],
        # Provenance from first shard (all should match)
        "parquet_sha256": data.get("parquet_sha256"),
        "teacher_model": data.get("teacher_model"),
        "image_size": data.get("image_size"),
    }


@dataclass
class Config:
    shards_dir: Path
    out: Path | None = None
    workers: int = 32


def main(cfg: Config) -> None:
    log.info("=" * 60)
    log.info("Scan Failed Indices")
    log.info("=" * 60)
    log.info(f"shards_dir: {cfg.shards_dir}")
    log.info(f"out: {cfg.out or '(auto)'}")
    log.info(f"workers: {cfg.workers}")

    assert cfg.shards_dir.exists(), f"Directory not found: {cfg.shards_dir}"
    paths = sorted(cfg.shards_dir.glob("*.pt"))
    assert paths, f"No .pt files in {cfg.shards_dir}"

    log.info(f"Found {len(paths)} shard files")
    log.info(f"First: {paths[0].name}, Last: {paths[-1].name}")

    records: list[dict] = []
    total_images = 0
    shards_with_failures = 0
    errors = 0
    provenance: dict | None = None
    t0 = time.perf_counter()

    log.info(f"Starting parallel scan with {cfg.workers} workers...")

    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        futures = {ex.submit(extract_failed, p): p for p in paths}
        pbar = tqdm(as_completed(futures), total=len(paths), desc="Shards", unit="shard")

        for fut in pbar:
            path = futures[fut]
            try:
                result = fut.result()
                shard_id = result["shard_id"]
                start_idx = result["start_idx"]
                shard_size = result["shard_size"]
                local_failed = result["failed_indices"]
                failed_paths = result["failed_paths"]

                total_images += shard_size

                # Capture provenance from first shard
                if provenance is None:
                    provenance = {
                        "parquet_sha256": result["parquet_sha256"],
                        "teacher_model": result["teacher_model"],
                        "image_size": result["image_size"],
                    }

                if local_failed:
                    shards_with_failures += 1
                    log.debug(
                        f"Shard {shard_id:05d}: {len(local_failed)} failed "
                        f"(start={start_idx}, size={shard_size})"
                    )

                for local_idx, path_str in zip(local_failed, failed_paths):
                    records.append({
                        "global_idx": start_idx + local_idx,
                        "shard_id": shard_id,
                        "local_idx": local_idx,
                        "path": path_str,
                    })

                pbar.set_postfix({
                    "failed": len(records),
                    "errors": errors,
                })

            except Exception as e:
                errors += 1
                log.error(f"FAILED to read {path.name}: {e}")

    elapsed = time.perf_counter() - t0
    log.info(f"Scan complete in {elapsed:.1f}s ({len(paths) / elapsed:.1f} shards/s)")
    log.info(f"Total images scanned: {total_images:,}")
    log.info(f"Total failed: {len(records):,} ({100 * len(records) / max(1, total_images):.4f}%)")
    log.info(f"Shards with failures: {shards_with_failures}/{len(paths)}")
    log.info(f"Read errors: {errors}")

    if provenance:
        log.info(f"Provenance: {provenance}")

    df = pl.DataFrame(records).sort("global_idx")

    # Add metadata columns
    now = datetime.now(UTC).isoformat()
    git_commit = get_git_commit()
    df = df.with_columns(
        pl.lit(str(cfg.shards_dir)).alias("shards_dir"),
        pl.lit(now).alias("scanned_at"),
        pl.lit(git_commit).alias("git_commit"),
        pl.lit(provenance["parquet_sha256"] if provenance else None).alias("parquet_sha256"),
        pl.lit(provenance["teacher_model"] if provenance else None).alias("teacher_model"),
        pl.lit(provenance["image_size"] if provenance else None).alias("image_size"),
    )

    out = cfg.out or cfg.shards_dir / "failed_index.parquet"

    log.info(f"Writing {len(df)} records to {out}")
    df.write_parquet(out)
    log.info(f"Output size: {out.stat().st_size / 1024:.1f} KB")
    log.info(f"Columns: {df.columns}")

    if len(df) > 0:
        log.info(f"\nFirst 10 failed:\n{df.select(['global_idx', 'shard_id', 'path']).head(10)}")

    log.info("Done.")


if __name__ == "__main__":
    main(tyro.cli(Config))
