"""The shard format, written by features.export and read by pretraining. Shards on disk use exactly these keys."""

from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor

from canvit_pytorch.teacher import DINOV3_REPOS

STORAGE_DTYPE = torch.float16

# The paper's ImageNet-21k shards (exported 2026-01-11) name their teacher by DINOv3's own checkpoint, as
# teacher_model, instead of teacher_repo_id; the Hub repo holds the same model.
TEACHER_REPOS_BY_MODEL = {"dinov3_vitb16": DINOV3_REPOS["vitb16"]}

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
    shard = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if "teacher_repo_id" not in shard:
        model = shard["teacher_model"]
        assert model in TEACHER_REPOS_BY_MODEL, f"{path}: teacher {model!r} has no known Hub repo"
        shard["teacher_repo_id"] = TEACHER_REPOS_BY_MODEL[model]
    return shard
