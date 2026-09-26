"""Tests for IndexedImageFolder."""

from pathlib import Path

import pytest

from .indexed_image_folder import SCHEMA_VERSION, IndexedImageFolder, IndexMetadata


def test_index_metadata_fields() -> None:
    meta = IndexMetadata(
        schema_version=SCHEMA_VERSION,
        root_name="test",
        n_samples=100,
        n_classes=10,
        generated_at="2025-01-01T00:00:00Z",
    )
    assert meta.schema_version == SCHEMA_VERSION
    assert meta.n_samples == 100


def test_creates_index(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    for i in range(3):
        cls_dir = root / f"class_{i}"
        cls_dir.mkdir()
        for j in range(5):
            (cls_dir / f"img_{j}.JPEG").write_bytes(b"fake")

    index_dir = tmp_path / "indices"
    ds = IndexedImageFolder(root, index_dir=index_dir)

    assert len(ds) == 15
    assert len(ds.classes) == 3
    assert (index_dir / "dataset.parquet").exists()


def test_loads_existing_index(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "c0").mkdir()
    (root / "c0" / "a.JPEG").write_bytes(b"x")

    index_dir = tmp_path / "indices"

    ds1 = IndexedImageFolder(root, index_dir=index_dir)
    assert len(ds1) == 1

    ds2 = IndexedImageFolder(root, index_dir=index_dir)
    assert len(ds2) == 1
    assert ds2.samples == ds1.samples


def test_root_mismatch_raises(tmp_path: Path) -> None:
    root1 = tmp_path / "dataset1"
    root1.mkdir()
    (root1 / "c").mkdir()
    (root1 / "c" / "x.JPEG").write_bytes(b"")

    index_dir = tmp_path / "indices"
    IndexedImageFolder(root1, index_dir=index_dir)

    root2 = tmp_path / "dataset2"
    root2.mkdir()
    (root2 / "d").mkdir()
    (root2 / "d" / "y.JPEG").write_bytes(b"")

    index_file = index_dir / "dataset1.parquet"
    wrong_index = index_dir / "dataset2.parquet"
    wrong_index.write_bytes(index_file.read_bytes())

    with pytest.raises(AssertionError, match="Root mismatch"):
        IndexedImageFolder(root2, index_dir=index_dir)
