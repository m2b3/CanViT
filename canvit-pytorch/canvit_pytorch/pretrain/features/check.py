"""Gate a shard tree before training on it: contiguous shards, the expected image count, no failed images
(their features are NaN and would train silently), and the same index, teacher and format in every shard.

    python -m canvit_pytorch.pretrain.features.check --shards-dir $FEATURES_DIR/in1k/dinov3_vitb16/512/shards \\
        --expected-images 1281167
"""

from dataclasses import dataclass
from pathlib import Path

import tyro

from canvit_pytorch.pretrain.features.shard import COMPATIBILITY_FIELDS, load_shard


@dataclass(frozen=True)
class Arguments:
    shards_dir: Path
    expected_images: int


def main(args: Arguments) -> None:
    paths = sorted(args.shards_dir.glob("*.pt"))
    assert paths, f"no shards in {args.shards_dir}"
    shard_ids: list[int] = []
    n_images = 0
    failed: dict[str, list[int]] = {}
    reference: dict[str, object] | None = None
    for path in paths:
        shard = load_shard(path)
        shard_ids.append(shard["shard_id"])
        assert len(shard["paths"]) == shard["end_idx"] - shard["start_idx"], f"{path.name}: paths do not fill its row range"
        n_images += len(shard["paths"])
        if shard["failed_indices"]:
            failed[path.name] = shard["failed_indices"]
        fields: dict[str, object] = {k: shard[k] for k in COMPATIBILITY_FIELDS}
        reference = reference or fields
        assert fields == reference, f"{path.name}: {fields} differs from {reference}"
    print(f"{len(paths)} shards, {n_images:,} images, format {reference}")
    assert shard_ids == list(range(len(paths))), "shard ids are not contiguous from 0"
    assert n_images == args.expected_images, f"{n_images:,} images, expected {args.expected_images:,}"
    assert not failed, f"failed images (NaN features): {failed}"
    print("PASS")


if __name__ == "__main__":
    main(tyro.cli(Arguments))
