"""Read images directly from mmap'd tar files. No extraction needed.

Scan tar headers to build a {name: (offset, size)} index, then read images
via mmap slicing. Forked DataLoader workers share mmap'd pages (copy-on-write).

scan_tar_headers(tar_path) — slow header scan; for export/bench.
load_tar_index(tar_path)   — instant load from .idx file; for training.

.idx files are produced by sa1b/build_tar_indexes.py with SHA256 + size.
"""

import io
import logging
import mmap
import pickle
import tarfile
import time
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

# {stripped_name: (data_offset, data_size)}
TarIndex = dict[str, tuple[int, int]]


def scan_tar_headers(tar_path: Path) -> TarIndex:
    """Scan tar headers → {stripped_name: (data_offset, data_size)}.

    Slow on large tars. Use for export scripts and benchmarks.
    For training, use load_tar_index() instead.
    """
    t0 = time.perf_counter()
    index: TarIndex = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.name.endswith(".jpg"):
                continue
            stripped = member.name.split("/", 1)[-1] if "/" in member.name else member.name
            index[stripped] = (member.offset_data, member.size)
    elapsed = time.perf_counter() - t0
    log.info(f"Scanned tar: {tar_path.name}, {len(index)} JPEGs in {elapsed:.1f}s")
    return index


def load_tar_index(tar_path: Path) -> TarIndex:
    """Load pre-built .idx file for a tar. Crashes if missing or stale.

    .idx files are built by sa1b/build_tar_indexes.py. Verifies tar file
    size matches (instant stat() check, no full read).
    """
    idx_path = tar_path.parent / f"{tar_path.name}.idx"
    assert idx_path.exists(), (
        f"No .idx for {tar_path.name}. "
        f"Run: uv run python sa1b/build_tar_indexes.py --tar-dir {tar_path.parent}"
    )

    t0 = time.perf_counter()
    with open(idx_path, "rb") as f:
        data = pickle.load(f)

    actual_size = tar_path.stat().st_size
    assert data["tar_size"] == actual_size, (
        f"Tar size mismatch: {tar_path.name} "
        f"(index={data['tar_size']}, actual={actual_size}). "
        f"Re-run: uv run python sa1b/build_tar_indexes.py --tar-dir {tar_path.parent} --force"
    )

    index = data["index"]
    elapsed = time.perf_counter() - t0
    log.info(
        f"Loaded tar index: {tar_path.name}, {len(index)} JPEGs "
        f"(sha256={data['sha256'][:12]}..., {elapsed:.3f}s)"
    )
    return index


class TarImageReader:
    """Read images from an mmap'd tar file by name."""

    def __init__(self, tar_path: Path, *, index: TarIndex) -> None:
        self._fd = open(tar_path, "rb")
        self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)
        self.index = index

    def read_image(self, name: str) -> Image.Image:
        assert self._mm is not None
        data_offset, size = self.index[name]
        return Image.open(io.BytesIO(self._mm[data_offset : data_offset + size])).convert("RGB")

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None  # type: ignore[assignment]
        if self._fd is not None:
            self._fd.close()
            self._fd = None  # type: ignore[assignment]

    def __del__(self) -> None:
        self.close()
