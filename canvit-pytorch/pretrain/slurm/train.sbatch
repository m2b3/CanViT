#!/bin/bash
#SBATCH --job-name=canvit-in21k-pretrain
#SBATCH --time=00:45:00
#SBATCH --array=0-400%1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/train-%A_%a.out
#SBATCH --error=logs/train-%A_%a.err
#SBATCH --mail-user=me@yberreby.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

# CanViT pretraining: 401 jobs × 4992 steps ≈ 2M steps (~8 days)
#
# Default (CanViT-B):
#   sbatch slurm/train.sbatch
#
# If your cluster requires an allocation account:
#   sbatch --account=my_project_name slurm/train.sbatch
#
# CanViT-S:
#   sbatch slurm/train.sbatch --backbone-name vits16
#
# Continue existing run:
#   sbatch --array=0-99%1 slurm/train.sbatch --run-name train-XXXXXXX
#
# Quick test:
#   sbatch --array=0-2%1 --time=00:10:00 slurm/train.sbatch --steps-per-job 10

set -eu  # NOT -x: would trace secret exports (HF_TOKEN, COMET_API_KEY) into logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }
timed() { local start=$SECONDS; "$@"; log "  ↳ $((SECONDS - start))s"; }

log "=== CanViT Training ==="
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    log "Array Job: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID, Host: $(hostname)"
    RUN_NAME="train-${SLURM_ARRAY_JOB_ID}"
else
    log "Job: $SLURM_JOB_ID on $(hostname)"
    RUN_NAME=""
fi
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm/env.sh

# Dependencies
log "Installing dependencies..."
timed uv sync

# Ensure logs dir exists
mkdir -p logs

# Training with precomputed features (teacher loaded from HF Hub).
# DATASET selects the feature-shard tree and is recorded in checkpoint
# metadata; FEATURE_IMAGE_ROOT must point at the matching image tree when
# DATASET != in21k. Both come from the submission environment, e.g.:
#   DATASET=in1k FEATURE_IMAGE_ROOT=/path/to/in1k/train sbatch slurm/train.sbatch
log "Starting training..."
DATASET=${DATASET:-in21k}
RUN_NAME_ARG=()
[ -n "$RUN_NAME" ] && RUN_NAME_ARG=(--run-name "$RUN_NAME")
exec uv run python -m canvit_pretrain.train \
    "${RUN_NAME_ARG[@]}" \
    --dataset "$DATASET" \
    --feature-base-dir "$FEATURES_DIR/$DATASET" \
    --feature-image-root "${FEATURE_IMAGE_ROOT:-$IN21K_IMAGE_DIR}" \
    --val-dir "$IN1K_VAL_IMAGE_DIR" \
    --train-index-dir "$INDEX_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    "$@"
