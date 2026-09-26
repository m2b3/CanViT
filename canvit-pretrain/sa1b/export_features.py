"""Export DINOv3 features for one SA-1B tar.

Designed for SLURM array jobs: 1 task = 1 tar = 1 shard.

Each invocation:
  1. Reads images directly from the tar via mmap (no extraction)
  2. Loads frozen DINOv3 teacher from HuggingFace Hub
  3. Runs batched inference on all JPEGs
  4. Saves one .pt shard (named after tar: sa_NNNNNN.pt)

Images are iterated in TAR FILE ORDER (not alphabetical). This is critical:
the training loader iterates the shard sequentially, so shard order must match
tar order for sequential I/O during training.

Shard format matches training loader expectations (shards.py):
  patches:        [N, n_patches, embed_dim] float16
  cls:            [N, embed_dim] float32
  paths:          list[str] — filenames in TAR ORDER
  class_idxs:     [N] int32 (all 0 for SA-1B)
  failed_indices: list[int]
  image_hashes:   list[str] — xxh64 of decoded pixels

Usage:
  # Single tar (interactive)
  uv run python sa1b/export_features.py \
      --tar /path/to/sa_000020.tar \
      --out-dir /path/to/shards

  # SLURM array job (see sa1b/export_features.sh)
  sbatch --array=0-999 sa1b/export_features.sh
"""

import logging
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import tyro
import xxhash
from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.teacher import load_teacher
from PIL import ImageFile
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from canvit_pretrain.train.data.tar_images import TarImageReader, scan_tar_headers

STORAGE_DTYPE = torch.float16
NUMPY_DTYPE = np.float16  # Must match STORAGE_DTYPE
ImageFile.LOAD_TRUNCATED_IMAGES = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


@dataclass
class Config:
    tar: Path
    out_dir: Path
    tmp_dir: Path = Path("/tmp")  # For mmap accumulation buffers
    image_size: int = 1024
    teacher_repo_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    batch_size: int = 32
    num_workers: int = 4


class TarImageDataset(Dataset[tuple[Tensor, int, bool, str]]):
    """Map-style dataset reading images from an mmap'd tar file.

    Images are indexed in tar file order (not alphabetical).
    DataLoader workers share the mmap via fork COW.
    """

    def __init__(self, reader: TarImageReader, size: int) -> None:
        self.reader = reader
        # Dict preserves insertion order (Python 3.7+) = tar file order
        self.names = list(reader.index.keys())
        self.transform = preprocess(size)
        self.size = size

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, bool, str]:
        name = self.names[idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                img = self.reader.read_image(name)
                img.load()
            img_hash = xxhash.xxh64(img.tobytes()).hexdigest()
            tensor = self.transform(img)
            assert isinstance(tensor, Tensor)
            return tensor, idx, True, img_hash
        except Exception as e:
            log.warning(f"Bad image {name}: {e}")
            return torch.full((3, self.size, self.size), float("nan")), idx, False, ""


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def main(cfg: Config) -> None:
    t_start = time.perf_counter()
    device = torch.device("cuda")
    tar_stem = cfg.tar.stem  # e.g. "sa_000020"
    shard_path = cfg.out_dir / f"{tar_stem}.pt"

    # Skip if already exported
    if shard_path.exists():
        log.info(f"Shard already exists: {shard_path}")
        return

    assert cfg.tar.exists(), f"Tar not found: {cfg.tar}"

    log.info(f"tar: {cfg.tar}")
    log.info(f"out_dir: {cfg.out_dir}")
    log.info(f"image_size: {cfg.image_size}")
    log.info(f"teacher_repo_id: {cfg.teacher_repo_id}")

    # --- Phase 1: Index tar (no extraction) ---
    # ~44s on cold NFS, ~2s warm cache (70 GB tar, 11k JPEGs)
    log.info(f"Indexing {cfg.tar.name} via mmap...")
    t0 = time.perf_counter()
    index = scan_tar_headers(cfg.tar)
    reader = TarImageReader(cfg.tar, index=index)
    n = len(reader.index)
    assert n > 0, f"No JPEGs in {cfg.tar}"
    t_index = time.perf_counter() - t0
    log.info(f"Index: {n} JPEGs in {t_index:.1f}s")

    # --- Phase 2: Load teacher ---
    t0 = time.perf_counter()
    teacher = load_teacher(cfg.teacher_repo_id, device)
    patch_size = teacher.model.config.patch_size
    embed_dim = teacher.embed_dim
    n_patches = (cfg.image_size // patch_size) ** 2
    assert cfg.image_size % patch_size == 0
    t_teacher = time.perf_counter() - t0
    log.info(f"Teacher: {embed_dim}d, {n_patches} patches, patch_size={patch_size} (loaded in {t_teacher:.1f}s)")
    log.info(f"GPU after teacher: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # --- Phase 3: Inference ---
    # ~195s for 11k images @ 1024px on H100 (~57 img/s)
    # Accumulate into mmap'd files — the full shard is ~66 GB in fp16,
    # won't fit in RAM. mmap lets the OS page to/from SSD as needed.
    scratch = Path(cfg.tmp_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    patches_mmap_path = scratch / f"{tar_stem}_patches.mmap"
    cls_mmap_path = scratch / f"{tar_stem}_cls.mmap"
    patches_buf = np.memmap(patches_mmap_path, dtype=NUMPY_DTYPE, mode="w+", shape=(n, n_patches, embed_dim))
    cls_buf = np.memmap(cls_mmap_path, dtype=np.float32, mode="w+", shape=(n, embed_dim))
    log.info(f"Mmap buffers: patches={patches_buf.nbytes/1e9:.1f}GB cls={cls_buf.nbytes/1e6:.0f}MB")

    hashes: list[str] = [""] * n
    failed: list[int] = []

    dataset = TarImageDataset(reader, cfg.image_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # CRITICAL: preserve tar order
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    t0 = time.perf_counter()
    n_batches = 0
    t_data_total = 0.0
    t_gpu_total = 0.0
    write_idx = 0
    pbar = tqdm(loader, desc=tar_stem)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t_batch_end = time.perf_counter()
        for imgs, indices, ok, batch_hashes in pbar:
            t_data = time.perf_counter() - t_batch_end
            t_data_total += t_data

            for i, success, h in zip(indices.tolist(), ok.tolist(), batch_hashes):
                hashes[i] = h
                if not success:
                    failed.append(i)

            t_gpu_start = time.perf_counter()
            imgs = imgs.to(device, non_blocking=True)
            feats = teacher.forward_norm_features(imgs)
            bs = imgs.shape[0]
            # GPU → CPU → mmap (OS handles write-back to SSD asynchronously)
            patches_buf[write_idx : write_idx + bs] = feats.patches.to(STORAGE_DTYPE).cpu().numpy()
            cls_buf[write_idx : write_idx + bs] = feats.cls.float().cpu().numpy()
            t_gpu = time.perf_counter() - t_gpu_start
            t_gpu_total += t_gpu

            write_idx += bs
            n_batches += 1
            pbar.set_postfix_str(f"data={t_data:.2f}s gpu={t_gpu:.2f}s img/s={write_idx/(time.perf_counter()-t0):.0f}")
            t_batch_end = time.perf_counter()

    # Sync before asserting — GPU ops may still be in flight
    torch.cuda.synchronize()
    assert write_idx == n, f"Expected {n}, wrote {write_idx}"
    patches_buf.flush()
    cls_buf.flush()
    t_inference = time.perf_counter() - t0

    log.info(
        f"Inference: {n} images in {t_inference:.1f}s ({n / t_inference:.0f} img/s) | "
        f"data: {t_data_total:.1f}s ({t_data_total / t_inference * 100:.0f}%) | "
        f"gpu: {t_gpu_total:.1f}s ({t_gpu_total / t_inference * 100:.0f}%)"
    )
    if failed:
        log.warning(f"{len(failed)} failed images")

    # --- Phase 4: Save ---
    # ~231s for 70 GB shard to NFS (~305 MB/s observed on Nibi)
    # Paths in tar order (dataset.names = tar iteration order)
    filenames = dataset.names
    shard_mb_est = (patches_buf.nbytes + cls_buf.nbytes) / 1e6
    log.info(f"Saving shard to {shard_path} (via .tmp, ~{shard_mb_est:.0f} MB)...")
    t0 = time.perf_counter()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    tmp = shard_path.with_suffix(".tmp")
    torch.save(
        {
            "patches": torch.from_numpy(patches_buf),
            "cls": torch.from_numpy(cls_buf),
            "paths": filenames,
            "class_idxs": torch.zeros(n, dtype=torch.int32),
            "image_hashes": hashes,
            "failed_indices": failed,
            # Metadata
            "tar_name": cfg.tar.name,
            "image_size": cfg.image_size,
            "teacher_repo_id": cfg.teacher_repo_id,
            "dtype": str(STORAGE_DTYPE),
            "embed_dim": embed_dim,
            "n_patches": n_patches,
            "n_images": n,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": get_git_commit(),
        },
        tmp,
    )
    tmp.rename(shard_path)
    t_save = time.perf_counter() - t0

    # Clean up mmap temp files (data is now in the .pt shard)
    del patches_buf, cls_buf
    patches_mmap_path.unlink(missing_ok=True)
    cls_mmap_path.unlink(missing_ok=True)

    reader.close()

    shard_mb = shard_path.stat().st_size / 1e6
    t_total = time.perf_counter() - t_start

    log.info(f"Saved {shard_path.name} ({shard_mb:.0f} MB, {n} images)")
    log.info(
        f"Timing: index={t_index:.0f}s teacher={t_teacher:.0f}s "
        f"inference={t_inference:.0f}s save={t_save:.0f}s total={t_total:.0f}s"
    )
    try:
        import resource
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB → MB
        peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e6
        log.info(f"Peak memory: GPU={peak_gpu_mb:.0f}MB CPU_RSS={peak_rss_mb:.0f}MB")
    except Exception:
        pass


if __name__ == "__main__":
    main(tyro.cli(Config))
