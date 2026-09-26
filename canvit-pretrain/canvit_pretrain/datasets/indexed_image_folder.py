"""Parquet-indexed ImageFolder dataset for large class-folder image trees.

First use scans the filesystem and saves a parquet index. Later runs load the
index directly, avoiding repeated metadata walks over millions of image files.
"""

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class IndexMetadata:
    schema_version: int
    root_name: str
    n_samples: int
    n_classes: int
    generated_at: str


class IndexedImageFolder(Dataset):
    """Dataset with automatic parquet indexing for fast loading."""

    classes: list[str]
    class_to_idx: dict[str, int]
    samples: list[tuple[str, int]]

    def __init__(
        self,
        root: Path,
        index_dir: Path,
        transform: Compose | None = None,
    ):
        self.root = root
        self.transform = transform
        self.loader = default_loader

        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / f"{root.name}.parquet"

        if index_path.exists():
            self._load_index(index_path)
        else:
            self._scan_and_save(index_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        path, target = self.samples[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def _load_index(self, index_path: Path) -> None:
        import pyarrow.parquet as pq

        log.info(f"Loading index: {index_path}")
        t0 = time.perf_counter()

        table = pq.read_table(index_path)
        meta = _parse_metadata(table.schema.metadata or {})

        assert meta.root_name == self.root.name, (
            f"Root mismatch: index has '{meta.root_name}', actual is '{self.root.name}'"
        )

        log.info(f"  {meta.n_samples:,} samples, {meta.n_classes:,} classes")

        paths = table.column("path").to_pylist()
        class_names = table.column("class_name").to_pylist()
        class_idxs = table.column("class_idx").to_pylist()

        self.classes = sorted(set(class_names))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = [(f"{self.root}/{p}", idx) for p, idx in zip(paths, class_idxs)]

        log.info(f"  ready in {time.perf_counter() - t0:.2f}s")

    def _scan_and_save(self, index_path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tqdm import tqdm

        log.info(f"Index not found, scanning: {self.root}")
        t0 = time.perf_counter()

        class_dirs = sorted(
            d for d in self.root.iterdir() if d.is_dir() and d.name != "tars"
        )
        log.info(f"  {len(class_dirs):,} classes")

        paths: list[str] = []
        class_names: list[str] = []
        for d in tqdm(class_dirs, desc="Scanning", unit="class"):
            cn = d.name
            for f in d.iterdir():
                if f.is_file():
                    paths.append(f"{cn}/{f.name}")
                    class_names.append(cn)

        n_samples = len(paths)
        assert n_samples > 0, f"No images found in {self.root}"
        log.info(f"  {n_samples:,} images")

        self.classes = sorted(set(class_names))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        class_idxs = [self.class_to_idx[cn] for cn in class_names]

        table = pa.table({
            "path": pa.array(paths, type=pa.string()),
            "class_name": pa.array(class_names, type=pa.string()),
            "class_idx": pa.array(class_idxs, type=pa.int32()),
        })
        table = table.replace_schema_metadata({
            b"schema_version": str(SCHEMA_VERSION).encode(),
            b"root_name": self.root.name.encode(),
            b"n_samples": str(n_samples).encode(),
            b"n_classes": str(len(self.classes)).encode(),
            b"generated_at": datetime.now(UTC).isoformat().encode(),
        })
        pq.write_table(table, index_path, compression="zstd")

        size_mb = index_path.stat().st_size / (1024 * 1024)
        log.info(f"  saved: {index_path.name} ({size_mb:.1f} MB)")

        self.samples = [(f"{self.root}/{p}", idx) for p, idx in zip(paths, class_idxs)]
        log.info(f"  complete in {time.perf_counter() - t0:.1f}s")


def _parse_metadata(raw: dict[bytes, bytes]) -> IndexMetadata:
    v = int(raw.get(b"schema_version", b"-1").decode())
    assert v == SCHEMA_VERSION, f"Schema mismatch: {v} != {SCHEMA_VERSION}"

    def get(k: str) -> str:
        return raw.get(k.encode(), b"").decode()

    return IndexMetadata(
        schema_version=v,
        root_name=get("root_name"),
        n_samples=int(get("n_samples") or 0),
        n_classes=int(get("n_classes") or 0),
        generated_at=get("generated_at"),
    )
