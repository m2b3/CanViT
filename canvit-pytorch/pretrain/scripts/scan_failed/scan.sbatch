#!/bin/bash
#SBATCH --job-name=scan-failed
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --output=logs/scan-failed-%j.out
#SBATCH --error=logs/scan-failed-%j.err

# Scan feature shards for failed indices
#
# Usage:
#   sbatch scripts/scan_failed/scan.sbatch
#   sbatch --account=my_project_name scripts/scan_failed/scan.sbatch
#   sbatch scripts/scan_failed/scan.sbatch --time=01:00:00  # more time if needed

set -eux

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Scan Failed Indices ==="
log "Job: $SLURM_JOB_ID on $(hostname), CPUs: $SLURM_CPUS_PER_TASK"

source slurm/env.sh

mkdir -p logs

log "Installing dependencies..."
uv sync

SHARDS_DIR="$FEATURES_DIR/in21k/dinov3_vitb16/512/shards"
OUT_FILE="$FEATURES_DIR/in21k/failed_index.parquet"

log "Scanning: $SHARDS_DIR"
log "Output: $OUT_FILE"

exec uv run python -m scripts.scan_failed \
    "$SHARDS_DIR" \
    --out "$OUT_FILE" \
    --workers "$SLURM_CPUS_PER_TASK"
