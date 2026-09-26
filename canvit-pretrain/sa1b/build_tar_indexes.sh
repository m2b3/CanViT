#!/bin/bash
#SBATCH --job-name=build-tar-idx
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/build-tar-idx-%j.out
#SBATCH --error=logs/build-tar-idx-%j.err

# Build .idx files for all SA-1B tars. CPU-only, no GPU.
# ~40s per tar (2s scan + 35s SHA), 130 tars / 8 workers ≈ 11 min.

set -eu  # NOT -x: env.sh may contain secrets

source slurm/env.sh
mkdir -p logs

exec uv run python sa1b/build_tar_indexes.py \
    --tar-dir "$SA1B_TAR_DIR" \
    --workers 8
