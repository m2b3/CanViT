"""Export teacher features to sharded .pt files.

Precomputes DINOv3 features for all images, stores as sharded .pt files.
This eliminates expensive teacher inference during training.

USAGE:
    # Single shard
    uv run python scripts/export_in21k_features.py \
        --parquet /path/to/index.parquet \
        --image-root /path/to/images \
        --out-dir /path/to/output \
        --teacher-repo-id facebook/dinov3-vitb16-pretrain-lvd1689m \
        --shard 0

    # Range of shards (for SLURM array jobs)
    uv run python scripts/export_in21k_features.py ... --start-shard 0 --end-shard 100

SHARD SCHEMA:
    # Data
    # 🔴 POST-LAYERNORM hidden states, NOT L2-normalized. This said
    # "L2-normalized" until 2026-07-26 and it was simply wrong:
    # `DINOv3Teacher.forward_norm_features` returns
    # `model(images).last_hidden_state` split into CLS and patches, with no
    # normalization step anywhere (read from
    # CanViT-PyTorch/canvit_pytorch/teacher/__init__.py, 2026-07-26 ~17:35 EDT;
    # that method's own docstring already said "post-norm"). Measured on shard
    # 00000 row 0: patch L2 = 12.13, CLS L2 = 17.31, against 1.0 for a unit
    # vector. Consumers standardize per position, so training is unaffected —
    # but any claim about feature scale or cosine similarity that assumes unit
    # norm is false, and one was written down before this was checked.
    patches: [N, n_patches, embed_dim] STORAGE_DTYPE  - patch features
    cls: [N, embed_dim] STORAGE_DTYPE                 - CLS token
    paths: list[str]                                  - relative paths within image_root
    class_idxs: [N] int32                             - class indices from parquet
    image_hashes: list[str]                           - xxh64 of decoded pixels (empty string if failed)
    failed_indices: list[int]                         - indices where load failed (NaN features)

    # Position
    shard_id, start_idx, end_idx

    # Compatibility (must match across all shards)
    parquet_path, parquet_sha256, teacher_repo_id,
    image_size, shard_size, dtype, embed_dim, n_patches

    # Provenance
    created_at, git_commit
"""

import gc
import hashlib
import logging
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path

import pyarrow.parquet as pq
import torch
import tyro
import xxhash
from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.teacher import load_teacher
from PIL import Image, ImageFile
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

STORAGE_DTYPE = torch.float16
STORAGE_BYTES = torch.tensor([], dtype=STORAGE_DTYPE).element_size()

ImageFile.LOAD_TRUNCATED_IMAGES = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class Config:
    """CLI arguments."""

    # Required - no defaults for anything affecting correctness
    parquet: Path
    image_root: Path
    out_dir: Path
    teacher_repo_id: str
    image_size: int

    # Shard selection
    shard: int | None = None
    start_shard: int | None = None
    end_shard: int | None = None

    # Operational - safe defaults
    shard_size: int = 4096
    batch_size: int = 64
    num_workers: int = 8


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def estimate_bytes(n_images: int, n_patches: int, embed_dim: int) -> int:
    """Estimate bytes for features only."""
    return n_images * (n_patches + 1) * embed_dim * STORAGE_BYTES


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class ImageDataset(Dataset):
    """Load and transform images. Returns (tensor, index, success, hash)."""

    def __init__(self, root: Path, paths: list[str], size: int):
        self.root = root
        self.paths = paths
        self.size = size
        self.transform = preprocess(size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, bool, str]:
        path = self.root / self.paths[idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with Image.open(path) as f:
                    img = f.convert("RGB")
                    img.load()
            # Hash decoded pixels (before transform) for integrity verification
            img_hash = xxhash.xxh64(img.tobytes()).hexdigest()
            tensor = self.transform(img)
            assert isinstance(tensor, Tensor)
            return tensor, idx, True, img_hash
        except Exception as e:
            log.warning(f"Bad image {path}: {e}")
            return torch.full((3, self.size, self.size), float("nan")), idx, False, ""


# -----------------------------------------------------------------------------
# Exporter
# -----------------------------------------------------------------------------


class FeatureExporter:
    """Exports teacher features to sharded .pt files.

    Holds all shared state: teacher model, config, hashes, etc.
    Call run() after construction.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda")

        # Log config upfront
        log.info(f"parquet: {cfg.parquet}")
        log.info(f"image_root: {cfg.image_root}")
        log.info(f"out_dir: {cfg.out_dir}")
        log.info(f"teacher_repo_id: {cfg.teacher_repo_id}")
        log.info(f"image_size: {cfg.image_size}")
        log.info(f"dtype: {STORAGE_DTYPE} ({STORAGE_BYTES} bytes)")

        # Preflight first (check files exist before reading)
        self._preflight()
        log.info("Preflight OK")

        self.shards_dir = cfg.out_dir / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        # Parquet metadata (cheap: schema only)
        self.n_images = pq.read_metadata(cfg.parquet).num_rows
        self.n_shards = ceil(self.n_images / cfg.shard_size)
        log.info(f"Parquet: {self.n_images:,} images → {self.n_shards} shards")

        # Check shard range early (before expensive teacher load)
        shard_range = self._parse_shard_range()
        if shard_range is None:
            log.info(f"Requested range beyond {self.n_shards} shards, nothing to do")
            sys.exit(0)
        self.start, self.end = shard_range

        # Provenance (computed ONCE, not per shard)
        self.parquet_hash = sha256_file(cfg.parquet)
        self.git_commit = get_git_commit()
        log.info(f"Parquet hash: {self.parquet_hash}")
        log.info(f"Git commit: {self.git_commit[:12]}")

        # Teacher (frozen DINOv3 from HuggingFace Hub)
        self.teacher = load_teacher(cfg.teacher_repo_id, self.device)

        patch_size = self.teacher.model.config.patch_size
        self.embed_dim = self.teacher.embed_dim
        self.n_patches = (cfg.image_size // patch_size) ** 2
        assert cfg.image_size % patch_size == 0, f"{cfg.image_size} % {patch_size} != 0"
        log.info(
            f"Teacher: {cfg.teacher_repo_id}, {self.embed_dim}d, {self.n_patches} patches"
        )
        self._log_gpu("after teacher")

    def _parse_shard_range(self) -> tuple[int, int] | None:
        """Returns (start, end) or None if range is empty (beyond actual shards)."""
        cfg = self.cfg
        if cfg.shard is not None:
            if cfg.shard >= self.n_shards:
                return None  # Beyond actual shards
            return cfg.shard, cfg.shard + 1
        if cfg.start_shard is not None and cfg.end_shard is not None:
            start = cfg.start_shard
            end = min(cfg.end_shard, self.n_shards)
            if start >= end:
                return None  # Beyond actual shards
            return start, end
        raise ValueError("Specify --shard or --start-shard/--end-shard")

    def _preflight(self) -> None:
        cfg = self.cfg
        assert cfg.parquet.exists(), f"Parquet not found: {cfg.parquet}"
        assert cfg.image_root.is_dir(), f"Not a directory: {cfg.image_root}"

        schema = pq.read_schema(cfg.parquet)
        missing = {"path", "class_idx"} - set(schema.names)
        assert not missing, f"Parquet missing columns: {missing}"

    def _log_gpu(self, label: str) -> None:
        log.info(f"GPU [{label}]: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    def _shard_image_count(self, shard_id: int) -> int:
        """Actual image count for a shard (handles last shard being smaller)."""
        start_idx = shard_id * self.cfg.shard_size
        end_idx = min(start_idx + self.cfg.shard_size, self.n_images)
        return end_idx - start_idx

    def run(self) -> None:
        """Export all shards in range, skipping existing ones."""
        cfg = self.cfg

        # Determine shards to process
        shards_todo: list[int] = []
        images_todo = 0
        for sid in range(self.start, self.end):
            if (self.shards_dir / f"{sid:05d}.pt").exists():
                continue
            shards_todo.append(sid)
            images_todo += self._shard_image_count(sid)

        if not shards_todo:
            log.info("All shards exist")
            return

        # Estimate uses actual image counts, not shard_size
        est_bytes = sum(
            estimate_bytes(self._shard_image_count(sid), self.n_patches, self.embed_dim)
            for sid in shards_todo
        )
        log.info(
            f"Exporting: {len(shards_todo)} shards, {images_todo:,} images, "
            f"~{est_bytes / 1e9:.1f} GB"
        )

        # Load parquet table
        table = pq.read_table(cfg.parquet)

        t0_total = time.perf_counter()
        total_bytes = 0
        total_failed = 0

        pbar = tqdm(total=images_todo, unit="img", desc="Export")

        for shard_id in shards_todo:
            start_idx = shard_id * cfg.shard_size
            n = self._shard_image_count(shard_id)

            slice_table = table.slice(start_idx, n)
            paths = slice_table.column("path").to_pylist()
            class_idxs = slice_table.column("class_idx").to_pylist()

            n_failed, shard_bytes = self._export_shard(
                shard_id, paths, class_idxs, start_idx, pbar
            )
            total_bytes += shard_bytes
            total_failed += n_failed

        pbar.close()
        elapsed = time.perf_counter() - t0_total

        log.info(
            f"Done: {images_todo:,} images, {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s"
        )
        log.info(f"  {images_todo / elapsed:.0f} img/s, {total_bytes / elapsed / 1e6:.0f} MB/s")
        if total_failed:
            log.warning(f"  {total_failed} failed images")
        self._log_gpu("final")

    def _export_shard(
        self,
        shard_id: int,
        paths: list[str],
        class_idxs: list[int],
        start_idx: int,
        pbar: tqdm,
    ) -> tuple[int, int]:
        """Export one shard. Returns (n_failed, shard_bytes)."""
        cfg = self.cfg
        n = len(paths)
        assert n == self._shard_image_count(shard_id), "Image count mismatch"

        shard_path = self.shards_dir / f"{shard_id:05d}.pt"

        # Preallocate GPU buffers
        patches_buf = torch.empty(
            n, self.n_patches, self.embed_dim, dtype=STORAGE_DTYPE, device=self.device
        )
        cls_buf = torch.empty(n, self.embed_dim, dtype=STORAGE_DTYPE, device=self.device)
        failed: list[int] = []
        hashes: list[str] = [""] * n  # Filled by index

        loader = DataLoader(
            ImageDataset(cfg.image_root, paths, cfg.image_size),
            batch_size=cfg.batch_size,
            shuffle=False,  # Deterministic order for reproducibility
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        t0 = time.perf_counter()
        write_idx = 0

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for imgs, indices, ok, batch_hashes in tqdm(loader, desc=f"Shard {shard_id}", leave=False):
                for i, success, h in zip(indices.tolist(), ok.tolist(), batch_hashes):
                    hashes[i] = h
                    if not success:
                        failed.append(i)

                imgs = imgs.to(self.device)
                feats = self.teacher.forward_norm_features(imgs)

                bs = imgs.shape[0]
                patches_buf[write_idx : write_idx + bs] = feats.patches.to(STORAGE_DTYPE)
                cls_buf[write_idx : write_idx + bs] = feats.cls.to(STORAGE_DTYPE)
                write_idx += bs

        assert write_idx == n, f"Expected {n}, wrote {write_idx}"

        # Atomic save
        tmp = shard_path.with_suffix(".tmp")
        torch.save(
            {
                # Data
                "patches": patches_buf,
                "cls": cls_buf,
                "paths": paths,
                "class_idxs": torch.tensor(class_idxs, dtype=torch.int32),
                "image_hashes": hashes,  # xxh64 of decoded pixels, for integrity
                "failed_indices": failed,
                # Position
                "shard_id": shard_id,
                "start_idx": start_idx,
                "end_idx": start_idx + n,
                # Compatibility
                "parquet_path": str(cfg.parquet),
                "parquet_sha256": self.parquet_hash,
                "teacher_repo_id": cfg.teacher_repo_id,
                "image_size": cfg.image_size,
                "shard_size": cfg.shard_size,
                "dtype": str(STORAGE_DTYPE),
                "embed_dim": self.embed_dim,
                "n_patches": self.n_patches,
                "batch_size": cfg.batch_size,
                # Provenance
                "created_at": datetime.now(UTC).isoformat(),
                "git_commit": self.git_commit,
            },
            tmp,
        )
        tmp.rename(shard_path)

        elapsed = time.perf_counter() - t0
        shard_bytes = shard_path.stat().st_size

        pbar.update(n)
        pbar.set_postfix({
            "img/s": f"{n / elapsed:.0f}",
            "MB/s": f"{shard_bytes / elapsed / 1e6:.0f}",
            "fail": len(failed),
        })

        # Free GPU memory
        del patches_buf, cls_buf, loader
        gc.collect()
        torch.cuda.empty_cache()

        return len(failed), shard_bytes


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(cfg: Config) -> None:
    FeatureExporter(cfg).run()


if __name__ == "__main__":
    main(tyro.cli(Config))
