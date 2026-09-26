"""List the images of a class-per-directory tree in a parquet index, shuffled so that every shard mixes classes.

Pretraining reads shards in order, so an unshuffled index would feed it one
class at a time. The shuffle seed is recorded in the index; an existing index
is never replaced, because shards record the hash of the index they came from.

    python -m canvit_pytorch.pretrain.features.index --image-root $IN21K_IMAGE_DIR --out $INDEX_DIR/in21k-shuffled.parquet
"""

import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

log = logging.getLogger(__name__)

INDEX_FORMAT = "canvit-image-index-4879965f-3849-4e59-b4fa-ee0943b8d0f1"
"""Columns path, class_name, class_idx; metadata as written by main."""


def scan(image_root: Path) -> pa.Table:
    """Columns path (relative to image_root), class_name, class_idx (index in sorted class names)."""
    class_names = sorted(d.name for d in image_root.iterdir() if d.is_dir())
    assert class_names, f"no class directories in {image_root}"
    paths: list[str] = []
    labels: list[int] = []
    for class_idx, name in enumerate(class_names):
        files = sorted(f.name for f in (image_root / name).iterdir() if f.is_file())
        paths += [f"{name}/{f}" for f in files]
        labels += [class_idx] * len(files)
        if class_idx % 1000 == 0:
            log.info(f"{class_idx}/{len(class_names)} classes, {len(paths):,} images")
    assert paths, f"no images in {image_root}"
    return pa.table({
        "path": pa.array(paths, type=pa.string()),
        "class_name": pa.array([p.split("/", 1)[0] for p in paths], type=pa.string()),
        "class_idx": pa.array(labels, type=pa.int32()),
    })


@dataclass(frozen=True)
class Arguments:
    image_root: Path
    out: Path
    shuffle: bool = True
    """Training exports need a shuffled index; an evaluation export, joined back by path, does not."""


def main(args: Arguments) -> None:
    assert not args.out.exists(), f"{args.out} exists; shards record its hash, so it is never replaced"
    table = scan(args.image_root)
    metadata = {
        "format": INDEX_FORMAT,
        "root_name": args.image_root.name,
        "n_samples": str(table.num_rows),
        "n_classes": str(len(set(table.column("class_idx").to_pylist()))),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    if args.shuffle:
        seed = secrets.randbits(63)
        table = table.take(pa.array(np.random.default_rng(seed).permutation(table.num_rows)))
        metadata["shuffle_seed"] = str(seed)
    table = table.replace_schema_metadata({k.encode(): v.encode() for k, v in metadata.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out, compression="zstd")
    log.info(f"Wrote {args.out}: {metadata}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(tyro.cli(Arguments))
