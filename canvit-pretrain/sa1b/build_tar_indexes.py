"""Pre-build tar indexes with SHA256 for integrity verification.

Scans tar headers and computes SHA256 for each tar file. Saves .idx files
next to the tars. Run on a CPU node before training — no GPU needed.

Usage:
    uv run python sa1b/build_tar_indexes.py --tar-dir /path/to/tars
    uv run python sa1b/build_tar_indexes.py --tar-dir /path/to/tars --workers 8
    uv run python sa1b/build_tar_indexes.py --tar-dir /path/to/tars --verify
    uv run python sa1b/build_tar_indexes.py --tar-dir /path/to/tars --force
"""

import hashlib
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from tqdm import tqdm

from canvit_pretrain.train.data.tar_images import scan_tar_headers


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _build_one(tar_path: Path, force: bool) -> tuple[str, int, float]:
    """Build .idx for one tar. Returns (name, n_images, elapsed_s)."""
    idx_path = tar_path.parent / f"{tar_path.name}.idx"

    if idx_path.exists() and not force:
        with open(idx_path, "rb") as f:
            data = pickle.load(f)
        return tar_path.name, data["n_images"], 0.0

    t0 = time.perf_counter()
    index = scan_tar_headers(tar_path)
    sha = _sha256(tar_path)
    tar_size = tar_path.stat().st_size

    data = {
        "version": 1,
        "tar_name": tar_path.name,
        "tar_size": tar_size,
        "sha256": sha,
        "n_images": len(index),
        "index": index,
    }

    # Atomic write (rename is atomic on POSIX)
    tmp = idx_path.parent / f".{idx_path.name}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(idx_path)

    return tar_path.name, len(index), time.perf_counter() - t0


def _verify_one(tar_path: Path) -> str:
    """Verify .idx for one tar. Returns status line."""
    idx_path = tar_path.parent / f"{tar_path.name}.idx"
    if not idx_path.exists():
        return f"MISSING  {tar_path.name}"

    with open(idx_path, "rb") as f:
        data = pickle.load(f)

    actual_size = tar_path.stat().st_size
    if data["tar_size"] != actual_size:
        return f"SIZE_ERR {tar_path.name} (idx={data['tar_size']}, actual={actual_size})"

    actual_sha = _sha256(tar_path)
    if data["sha256"] != actual_sha:
        return f"SHA_ERR  {tar_path.name} (idx={data['sha256'][:12]}..., actual={actual_sha[:12]}...)"

    return f"OK       {tar_path.name} ({data['n_images']} imgs, sha256={data['sha256'][:12]}...)"


@dataclass
class Args:
    tar_dir: Path
    workers: int = 4
    force: bool = False
    verify: bool = False
    tars: list[str] = field(default_factory=list)


def main() -> None:
    args = tyro.cli(Args)
    assert args.tar_dir.is_dir(), f"Not a directory: {args.tar_dir}"

    if args.tars:
        tars = [args.tar_dir / t for t in args.tars]
        for t in tars:
            assert t.exists(), f"Not found: {t}"
    else:
        tars = sorted(args.tar_dir.glob("*.tar"))
    assert tars, f"No .tar files in {args.tar_dir}"
    print(f"{len(tars)} tars in {args.tar_dir}")

    if args.verify:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_verify_one, t): t for t in tars}
            for fut in tqdm(as_completed(futures), total=len(tars), desc="Verifying"):
                tqdm.write(f"  {fut.result()}")
        return

    built = 0
    skipped = 0
    t_total = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_build_one, t, args.force): t for t in tars}
        for fut in tqdm(as_completed(futures), total=len(tars), desc="Building"):
            name, n_images, elapsed = fut.result()
            if elapsed > 0:
                built += 1
                tqdm.write(f"  {name}: {n_images} imgs in {elapsed:.1f}s")
            else:
                skipped += 1

    elapsed_total = time.perf_counter() - t_total
    print(f"\nDone in {elapsed_total:.0f}s: {built} built, {skipped} skipped")


if __name__ == "__main__":
    main()
