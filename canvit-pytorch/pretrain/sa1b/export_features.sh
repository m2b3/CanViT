#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:20:00
#SBATCH --output=logs/export_sa1b_%A_%a.out
#SBATCH --error=logs/export_sa1b_%A_%a.err

# ==============================================================================
# Export DINOv3 features for SA-1B. Array job: 1 task = 1 tar = 1 shard.
# ~8 min/tar on H100 (index=44s, inference=195s, save=231s for 70GB shard).
#
# USAGE:
#   sbatch --array=0-999 sa1b/export_features.sh          # all 1000 tars
#   sbatch --account=my_project_name --array=0-999 sa1b/export_features.sh
#   sbatch --array=0-2   sa1b/export_features.sh          # first 3 tars
#   sbatch --array=20    sa1b/export_features.sh          # single tar
# ==============================================================================

set -euo pipefail

source slurm/env.sh
mkdir -p logs

TAR_IDX=$(printf "%06d" "$SLURM_ARRAY_TASK_ID")
TAR_PATH="$SA1B_TAR_DIR/sa_${TAR_IDX}.tar"
IMAGE_SIZE=1024
OUT_DIR="$SA1B_FEATURES_DIR/sa1b/dinov3_vitb16/${IMAGE_SIZE}/shards"
TMPDIR="$SLURM_TMPDIR/export_tmp"

echo "========================================"
echo "SLURM_JOB_ID:       $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Node:               $(hostname)"
echo "Date:               $(date -Iseconds)"
echo "Git commit:         $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "========================================"
echo "TAR:                $TAR_PATH"
echo "OUT_DIR:            $OUT_DIR"
echo "IMAGE_SIZE:         $IMAGE_SIZE"
echo "========================================"

[[ -f "$TAR_PATH" ]] || { echo "FATAL: Tar not found: $TAR_PATH" >&2; exit 1; }

time uv run python sa1b/export_features.py \
    --tar "$TAR_PATH" \
    --out-dir "$OUT_DIR" \
    --tmp-dir "$TMPDIR" \
    --image-size "$IMAGE_SIZE"

echo "========================================"
echo "End time:           $(date -Iseconds)"
echo "========================================"
