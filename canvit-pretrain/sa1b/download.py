"""Download SA-1B tarballs from Meta CDN and store UNCOMPRESSED.

Meta distributes .tar files that are actually gzipped (5% ratio on JPEGs — pure
waste). This script downloads and decompresses in a single pipe, storing plain
tars on NFS. Plain tars extract at NFS speed (GB/s) vs 20 MB/s for gzipped.

Already-downloaded gzipped tars (from earlier runs) are detected and decompressed
in place.

Idempotent: skips tars that already exist and are uncompressed.
Safe to ctrl-C and rerun (partial downloads use .downloading suffix).
No resume on partial downloads (pipe through gunzip prevents wget -c).

Each tar is ~11 GB uncompressed, ~11k JPEGs + masks.
~50 MB/s on Nibi login nodes → ~3.5 min per tar.

Usage (from repo root):
    uv run python sa1b/download.py                # all 1000 tars
    uv run python sa1b/download.py --limit 3      # first 3 only
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

# sa_NNNNNN.tar — the actual image tars. The TSV also contains
# sa_images_ids.txt (metadata) and sa_co_gold.tar (benchmark), which we skip.
IMAGE_TAR_RE = re.compile(r"^sa_\d{6}\.tar$")

GZIP_MAGIC = b"\x1f\x8b"


def is_gzipped(path: Path) -> bool:
    """Check if a file starts with gzip magic bytes."""
    with open(path, "rb") as f:
        return f.read(2) == GZIP_MAGIC


def decompress_in_place(path: Path) -> None:
    """Decompress a gzipped file in place: read gzipped, write plain tar."""
    tmp = path.with_suffix(".decompressing")
    result = subprocess.run(
        f"gzip -dc '{path}' > '{tmp}'",
        shell=True,
        check=False,
    )
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        print(f"[FAILED] decompress {path.name}", file=sys.stderr)
        return
    size_before = path.stat().st_size
    size_after = tmp.stat().st_size
    tmp.rename(path)
    print(
        f"[decompressed] {path.name}: "
        f"{size_before / 1e9:.1f} GB -> {size_after / 1e9:.1f} GB"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download SA-1B tarballs from Meta CDN (stored uncompressed)"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max tars to process (0 = all)"
    )
    parser.add_argument(
        "--tar-dir", type=Path, default=None, help="Override $SA1B_TAR_DIR"
    )
    parser.add_argument(
        "--links", type=Path, default=None, help="Override $SA1B_LINKS"
    )
    args = parser.parse_args()

    tar_dir = args.tar_dir or Path(os.environ.get("SA1B_TAR_DIR", ""))
    links = args.links or Path(os.environ.get("SA1B_LINKS", ""))
    assert tar_dir != Path(""), "Set --tar-dir or $SA1B_TAR_DIR"
    assert links != Path(""), "Set --links or $SA1B_LINKS"
    assert links.is_file(), f"Links file not found: {links}"

    tar_dir.mkdir(parents=True, exist_ok=True)

    # sa1b_links.tsv: TSV with header "file_name\tcdn_link"
    # Contains 1000 image tars + 2 non-image files (sa_images_ids.txt, sa_co_gold.tar)
    with open(links) as f:
        reader = csv.DictReader(f, delimiter="\t")
        assert reader.fieldnames == [
            "file_name",
            "cdn_link",
        ], f"Unexpected columns: {reader.fieldnames}"
        rows = [r for r in reader if IMAGE_TAR_RE.match(r["file_name"])]

    assert len(rows) == 1000, f"Expected 1000 image tars, got {len(rows)}"

    if args.limit > 0:
        rows = rows[: args.limit]

    print("=== SA-1B Download ===")
    print(f"TAR_DIR: {tar_dir}")
    print(f"LINKS:   {links} ({len(rows)} tars to process)")
    print("======================")

    # Also download sa_images_ids.txt (canonical list of all image IDs) if not present
    with open(links) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["file_name"] == "sa_images_ids.txt":
                ids_dest = tar_dir.parent / "sa_images_ids.txt"
                if not ids_dest.exists():
                    print(f"[download] sa_images_ids.txt -> {ids_dest}")
                    subprocess.run(
                        [
                            "wget",
                            "-c",
                            "-q",
                            "--show-progress",
                            "-O",
                            str(ids_dest),
                            r["cdn_link"],
                        ],
                        check=True,
                    )
                else:
                    print("[skip] sa_images_ids.txt (already exists)")
                break

    # --- Cleanup: remove orphaned temp files from interrupted runs ---
    for suffix in (".downloading", ".decompressing"):
        for orphan in tar_dir.glob(f"*{suffix}"):
            print(f"[cleanup] removing orphan {orphan.name}")
            orphan.unlink()

    # --- Pass 1: decompress any previously-downloaded gzipped tars ---
    n_decompressed = 0
    for row in rows:
        dest = tar_dir / row["file_name"]
        if dest.exists() and is_gzipped(dest):
            print(f"[decompress] {dest.name} is gzipped, decompressing...")
            decompress_in_place(dest)
            n_decompressed += 1
    if n_decompressed:
        print(f"[info] Decompressed {n_decompressed} previously-downloaded tars")

    # --- Pass 2: download missing tars (piped through gunzip → stored uncompressed) ---
    downloaded = 0
    skipped = 0

    for row in rows:
        filename = row["file_name"]
        url = row["cdn_link"]
        dest = tar_dir / filename
        tmp = dest.with_suffix(".downloading")

        if dest.exists():
            skipped += 1
            continue

        print(f"[download+decompress] {filename}")
        # Pipe wget through gunzip: download compressed, store uncompressed.
        # No resume possible (gzip stream can't seek), but at 50+ MB/s
        # re-downloading a single 11 GB tar takes ~3.5 min.
        result = subprocess.run(
            f"wget -q --show-progress -O - '{url}' | gzip -d > '{tmp}'",
            shell=True,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"[FAILED] {filename} (exit {result.returncode})", file=sys.stderr
            )
            tmp.unlink(missing_ok=True)
            break

        tmp.rename(dest)
        size_gb = dest.stat().st_size / 1e9
        downloaded += 1
        print(f"[done] {filename} ({size_gb:.1f} GB, uncompressed)")

    print(
        f"=== Complete: {downloaded} downloaded, {skipped} skipped, "
        f"{n_decompressed} decompressed ==="
    )


if __name__ == "__main__":
    main()
