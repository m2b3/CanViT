#!/bin/bash
#SBATCH --job-name=canvit-sa1b
#SBATCH --time=01:00:00
#SBATCH --array=0-99%1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --output=logs/sa1b-train-%A_%a.out
#SBATCH --error=logs/sa1b-train-%A_%a.err
#SBATCH --mail-user=me@yberreby.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

# SA-1B continual pretraining.
# Images are read directly from mmap'd tar files (no extraction step).
#
# First run (seed from HF Hub):
#   sbatch sa1b/train.sh
#   sbatch --account=my_project_name sa1b/train.sh
#
# Continue existing run:
#   sbatch --array=0-99%1 sa1b/train.sh --run-name sa1b-train-XXXXXXX
#
# Quick test (1 shard = 174 steps):
#   sbatch --array=0-0%1 --time=00:20:00 sa1b/train.sh --steps-per-job 174

set -eu  # NOT -x: would trace secret exports into logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# === CONFIG ===
BATCH_SIZE=64
NUM_WORKERS=12           # 12w + 70GB mmap shard. 256G gives page cache breathing room (128G caused 55% d%).
STEPS_PER_JOB=1218       # 7 shards × 174 batches/shard
HF_SEED="canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"

log "=== SA-1B Continual Pretraining ==="
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    log "Array Job: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID, Host: $(hostname)"
    RUN_NAME="sa1b-train-${SLURM_ARRAY_JOB_ID}"
else
    log "Job: $SLURM_JOB_ID on $(hostname)"
    RUN_NAME="sa1b-manual-$(date +%Y%m%d-%H%M)"
fi

# Override RUN_NAME and STEPS_PER_JOB if passed in "$@"
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    case "${ARGS[i]}" in
        --run-name)      RUN_NAME="${ARGS[i+1]}" ;;
        --steps-per-job) STEPS_PER_JOB="${ARGS[i+1]}" ;;
    esac
done

log "Run name: $RUN_NAME"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm/env.sh

mkdir -p logs

# === TRAINING ===
# Images are read directly from tar files via mmap (--tar-dir).
# No extraction step needed — Python handles everything.
#
# --hf-seed-ckpt is safe to always pass: ignored on RESUME (latest.pt takes priority).
# Normalizer auto-detects uninitialized standardizers — no need for --reset-normalizer.
log "Starting training..."
exec uv run python -m canvit_pretrain.train \
    --run-name "$RUN_NAME" \
    --hf-seed-ckpt "$HF_SEED" \
    --feature-base-dir "$SA1B_FEATURES_DIR/sa1b" \
    --tar-dir "$SA1B_TAR_DIR" \
    --val-dir "$IN1K_VAL_IMAGE_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --canvas-patch-grid-size 64 \
    --scene-resolution 1024 \
    --steps-per-job "$STEPS_PER_JOB" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --warmup-steps 2000 \
    --normalizer-max-samples 1024 \
    --dataset sa1b \
    --comet-project canvit-sa1b \
    --viz-every-n-vals 1 \
    --curve-every-n-vals 1 \
    "$@"
