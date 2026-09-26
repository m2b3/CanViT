#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --array=0-89%20
#SBATCH --output=logs/export_%A_%a.out
#SBATCH --error=logs/export_%A_%a.err

# Export DINOv3 ViT-B/16 features of 512 px scenes (canvit_pytorch.pretrain.features.export).
# Array task i writes shards [36 i, 36 (i + 1)); tasks past the last shard exit at once.
# Existing shards are skipped, so a failed or partial export resumes by resubmitting,
# for all tasks or some (sbatch --array=3,17 ...). Run from canvit-pytorch/.
#
#   sbatch slurm/pretrain/export_features.sh                          ImageNet-21k
#   DATASET=in1k IMAGE_ROOT=/path/to/in1k/train sbatch --array=0-8 slurm/pretrain/export_features.sh
#
# INDEX (default $INDEX_DIR/$DATASET-shuffled.parquet, from features.index) and
# OUT_DIR (default $FEATURES_DIR/$DATASET/dinov3_vitb16/512) can be overridden
# from the environment. Check a finished tree with features.check before training.

set -euo pipefail

mkdir -p logs
source slurm/env.sh

SHARDS_PER_TASK=36
DATASET=${DATASET:-in21k}
INDEX=${INDEX:-$INDEX_DIR/$DATASET-shuffled.parquet}
IMAGE_ROOT=${IMAGE_ROOT:-$IN21K_IMAGE_DIR}
OUT_DIR=${OUT_DIR:-$FEATURES_DIR/$DATASET/dinov3_vitb16/512}
FIRST_SHARD=$((SLURM_ARRAY_TASK_ID * SHARDS_PER_TASK))

echo "[$(date -Is)] task $SLURM_ARRAY_TASK_ID on $(hostname): $INDEX ($IMAGE_ROOT) -> $OUT_DIR, shards from $FIRST_SHARD"
exec uv run python -u -m canvit_pytorch.pretrain.features.export \
    --index "$INDEX" \
    --image-root "$IMAGE_ROOT" \
    --out-dir "$OUT_DIR" \
    --first-shard "$FIRST_SHARD" \
    --end-shard "$((FIRST_SHARD + SHARDS_PER_TASK))"
