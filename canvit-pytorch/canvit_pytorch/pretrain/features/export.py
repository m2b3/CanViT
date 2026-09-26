"""Compute the teacher's features of indexed images and write them as shards.

Shard i holds index rows [i × shard_size, (i + 1) × shard_size). Existing
shards are skipped and a shard appears only once complete (written to a
temporary file, then renamed), so an interrupted export resumes by rerunning.
A SLURM array splits the shard range across jobs (slurm/pretrain/export_features.sh).

    python -m canvit_pytorch.pretrain.features.export --index in21k-shuffled.parquet \\
        --image-root $IN21K_IMAGE_DIR --out-dir $FEATURES_DIR/in21k/dinov3_vitb16/512 --first-shard 0 --end-shard 36
"""

import hashlib
import logging
import math
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import torch
import tyro
import xxhash
from PIL import Image, ImageFile
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.pretrain.features.shard import STORAGE_DTYPE, FeatureShard, shard_path
from canvit_pytorch.provenance import provenance
from canvit_pytorch.teacher import TEACHER_REPO, DINOv3Teacher, load_teacher

log = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = False


class IndexedImages(Dataset[tuple[Tensor, int, bool, str]]):
    """(image, position, loaded, pixel hash); an image that fails to load comes back as NaN, flagged."""

    def __init__(self, image_root: Path, paths: list[str], size_px: int) -> None:
        self.image_root = image_root
        self.paths = paths
        self.size_px = size_px
        self.transform = preprocess(size_px)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[Tensor, int, bool, str]:
        path = self.image_root / self.paths[i]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # PIL reports corrupt EXIF and truncation as warnings
                with Image.open(path) as file:
                    image = file.convert("RGB")
            pixels_hash = xxhash.xxh64(image.tobytes()).hexdigest()
            tensor = self.transform(image)
            assert isinstance(tensor, Tensor)
            return tensor, i, True, pixels_hash
        except Exception as e:  # noqa: BLE001  any unreadable image is recorded as failed, logged and counted
            log.warning(f"Failed to load {path}: {e}")
            return torch.full((3, self.size_px, self.size_px), float("nan")), i, False, ""


def file_sha256_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 16):
            digest.update(chunk)
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class Arguments:
    index: Path
    """Parquet index with path and class_idx columns (features.index)."""
    image_root: Path
    out_dir: Path
    """Shards go to out_dir/shards."""
    first_shard: int
    end_shard: int
    """Exclusive; clipped to the number of shards the index fills."""
    image_size_px: int = 512
    teacher_repo: str = TEACHER_REPO
    shard_size: int = 4096
    batch_size: int = 64
    num_workers: int = 8
    device: str = "cuda"


def export_shard(
    args: Arguments, *, teacher: DINOv3Teacher, shard_id: int, paths: list[str], class_idxs: list[int], index_hash: str,
) -> None:
    device = torch.device(args.device)
    n_patches = (args.image_size_px // teacher.patch_size) ** 2
    patches = torch.empty(len(paths), n_patches, teacher.embed_dim, dtype=STORAGE_DTYPE, device=device)
    cls = torch.empty(len(paths), teacher.embed_dim, dtype=STORAGE_DTYPE, device=device)
    hashes = [""] * len(paths)
    failed: list[int] = []
    loader = DataLoader(
        IndexedImages(args.image_root, paths, args.image_size_px),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    written = 0
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for images, positions, loaded, batch_hashes in loader:
            for i, ok, pixels_hash in zip(positions.tolist(), loaded.tolist(), batch_hashes, strict=True):
                hashes[i] = pixels_hash
                if not ok:
                    failed.append(i)
            features = teacher(images.to(device))
            patches[written : written + len(images)] = features.patches.to(STORAGE_DTYPE)
            cls[written : written + len(images)] = features.cls.to(STORAGE_DTYPE)
            written += len(images)
    assert written == len(paths), (written, len(paths))

    start = shard_id * args.shard_size
    shard: FeatureShard = {
        "patches": patches, "cls": cls, "paths": paths,
        "class_idxs": torch.tensor(class_idxs, dtype=torch.int32),
        "image_hashes": hashes, "failed_indices": failed,
        "shard_id": shard_id, "start_idx": start, "end_idx": start + len(paths),
        "parquet_path": str(args.index), "parquet_sha256": index_hash,
        "teacher_repo_id": args.teacher_repo, "image_size": args.image_size_px, "shard_size": args.shard_size,
        "dtype": str(STORAGE_DTYPE), "embed_dim": teacher.embed_dim, "n_patches": n_patches,
        "batch_size": args.batch_size,
        "created_at": datetime.now(UTC).isoformat(), "git_commit": provenance()["git_commit"],
    }
    path = shard_path(args.out_dir / "shards", shard_id)
    tmp = path.with_suffix(".tmp")
    torch.save(shard, tmp)
    tmp.rename(path)
    log.info(f"Wrote {path}: {len(paths)} images, {len(failed)} failed")


def main(args: Arguments) -> None:
    columns = set(pq.read_schema(args.index).names)
    assert {"path", "class_idx"} <= columns, f"{args.index} has columns {sorted(columns)}"
    assert args.image_root.is_dir(), args.image_root
    n_images = pq.read_metadata(args.index).num_rows
    end = min(args.end_shard, math.ceil(n_images / args.shard_size))
    todo = [s for s in range(args.first_shard, end) if not shard_path(args.out_dir / "shards", s).exists()]
    log.info(f"{args.index}: {n_images:,} images; shards [{args.first_shard}, {end}) to write: {todo}")
    if not todo:
        return
    (args.out_dir / "shards").mkdir(parents=True, exist_ok=True)
    index_hash = file_sha256_prefix(args.index)
    table = pq.read_table(args.index, columns=["path", "class_idx"])
    teacher = load_teacher(args.teacher_repo, torch.device(args.device))
    assert args.image_size_px % teacher.patch_size == 0, (args.image_size_px, teacher.patch_size)
    for shard_id in todo:
        started = time.perf_counter()
        rows = table.slice(shard_id * args.shard_size, args.shard_size)
        export_shard(
            args, teacher=teacher, shard_id=shard_id, index_hash=index_hash,
            paths=rows.column("path").to_pylist(), class_idxs=rows.column("class_idx").to_pylist(),
        )
        log.info(f"Shard {shard_id}: {rows.num_rows / (time.perf_counter() - started):.0f} images/s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(tyro.cli(Arguments))
