"""The shard format, written by features.export and read by pretraining. Shards on disk use exactly these keys."""

from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor

STORAGE_DTYPE = torch.float16

# Fields that must agree across the shards of one tree.
COMPATIBILITY_FIELDS = ("parquet_sha256", "teacher_repo_id", "image_size", "shard_size", "dtype", "embed_dim", "n_patches")


class FeatureShard(TypedDict):
    patches: Tensor  # [N, n_patches, embed_dim] STORAGE_DTYPE, the teacher's outputs after its final LayerNorm
    cls: Tensor  # [N, embed_dim] STORAGE_DTYPE
    paths: list[str]  # relative to the image root
    class_idxs: Tensor  # [N] int32
    image_hashes: list[str]  # xxh64 of the decoded pixels; "" where loading failed
    failed_indices: list[int]  # images that failed to load; their features are NaN
    shard_id: int
    start_idx: int  # row range of the shard in the index
    end_idx: int
    parquet_path: str
    parquet_sha256: str  # first 16 hex digits
    teacher_repo_id: str
    image_size: int
    shard_size: int
    dtype: str
    embed_dim: int
    n_patches: int
    batch_size: int
    created_at: str
    git_commit: str | None


def shard_path(shards_dir: Path, shard_id: int) -> Path:
    return shards_dir / f"{shard_id:05d}.pt"


def load_shard(path: Path) -> FeatureShard:
    """Memory-mapped: a tensor row is read from disk when it is indexed."""
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
