"""Build a deterministically-shuffled parquet index of image paths for feature export.

`scripts/export_in21k_features.py` consumes `$INDEX_DIR/<dataset>-shuffled.parquet`
and processes rows in parquet order. Shuffling ensures each shard contains a
mixed-class sample so the sequential-shard training loader does not see class
clusters.

Idempotent: if the shuffled parquet already exists, exits without touching it
(the export script hashes the parquet as part of shard provenance, so we never
silently replace it).

Usage:
    uv run python scripts/build_shuffled_index.py \
        --image-root $IN21K_IMAGE_DIR \
        --index-dir $INDEX_DIR \
        --dataset in21k
"""

import secrets
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

from canvit_pretrain.datasets import IndexedImageFolder


@dataclass
class Config:
    image_root: Path
    index_dir: Path
    dataset: str


def main(cfg: Config) -> None:
    shuffled = cfg.index_dir / f"{cfg.dataset}-shuffled.parquet"
    if shuffled.exists():
        print(f"{shuffled} exists; remove it to rebuild.")
        return

    # Auto-build the canonical (class-sorted) parquet if missing. Cheap reload
    # on subsequent runs; first run on a 13M-image tree takes ~8 min.
    IndexedImageFolder(cfg.image_root, cfg.index_dir)
    canonical = cfg.index_dir / f"{cfg.image_root.name}.parquet"
    assert canonical.exists(), f"IndexedImageFolder did not produce {canonical}"

    seed = secrets.randbits(63)
    table = pq.read_table(canonical)
    perm = np.random.default_rng(seed).permutation(table.num_rows)
    shuffled_table = table.take(pa.array(perm))

    md = dict(table.schema.metadata or {})
    md[b"shuffle_seed"] = str(seed).encode()
    md[b"source_parquet"] = canonical.name.encode()
    shuffled_table = shuffled_table.replace_schema_metadata(md)

    pq.write_table(shuffled_table, shuffled, compression="zstd")
    print(f"Wrote {shuffled} ({shuffled_table.num_rows:,} rows, seed={seed})")


if __name__ == "__main__":
    main(tyro.cli(Config))
