"""Gate check for an exported shard tree: contiguous shard_ids, expected image
count, zero failed loads (failed images carry NaN features and would train
silently), consistent provenance across shards.

    uv run python scripts/check_shard_completeness.py \
        --shards-dir $FEATURES_DIR/in1k/dinov3_vitb16/512/shards \
        --expected-images 1281167
"""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    args = parser.parse_args()

    shard_paths = sorted(args.shards_dir.glob("*.pt"))
    assert shard_paths, f"no shards in {args.shards_dir}"

    total_images = 0
    total_failed = 0
    shard_ids: list[int] = []
    compat_keys = ("parquet_sha256", "teacher_repo_id", "image_size", "shard_size", "dtype", "embed_dim", "n_patches")
    compat_ref: dict[str, object] | None = None

    for p in shard_paths:
        shard = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        shard_ids.append(shard["shard_id"])
        n = len(shard["paths"])
        assert n == shard["end_idx"] - shard["start_idx"], f"{p.name}: paths/range mismatch"
        total_images += n
        failed = shard["failed_indices"]
        if failed:
            print(f"FAILED LOADS in {p.name}: {failed}")
            total_failed += len(failed)
        compat = {k: shard[k] for k in compat_keys}
        if compat_ref is None:
            compat_ref = compat
        else:
            assert compat == compat_ref, f"{p.name}: compat drift {compat} != {compat_ref}"

    assert compat_ref is not None
    n_shards = len(shard_paths)
    print(f"shards={n_shards} images={total_images:,} failed={total_failed}")
    print(f"compat: {compat_ref}")

    assert shard_ids == list(range(n_shards)), "shard_ids not contiguous from 0"
    assert total_images == args.expected_images, f"image count {total_images:,} != expected {args.expected_images:,}"
    assert total_failed == 0, f"{total_failed} failed image loads (NaN features)"
    print("GATE PASS")


if __name__ == "__main__":
    main()
