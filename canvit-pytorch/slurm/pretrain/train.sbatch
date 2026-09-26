#!/bin/bash
#SBATCH --job-name=canvit-pretrain
#SBATCH --time=00:45:00
#SBATCH --array=0-400%1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/pretrain-%A_%a.out
#SBATCH --error=logs/pretrain-%A_%a.err

# CanViT pretraining as a chain of array tasks, one at a time: each trains 4,992
# steps and saves; the next resumes from that checkpoint. 401 tasks cover the
# paper's 2M steps; tasks after the last step exit at once. Run from canvit-pytorch/.
#
#   sbatch slurm/pretrain/train.sbatch                           CanViT-B on ImageNet-21k
#   DATASET=in1k IMAGES_DIR=/path/to/in1k/train sbatch slurm/pretrain/train.sbatch
#   RUN_NAME=pretrain-1234567 sbatch slurm/pretrain/train.sbatch continue that run
#   bash slurm/pretrain/ablation.sh no-reads                     a paper ablation
#
# Extra arguments go to `python -m canvit_pytorch.pretrain` (see its --help).
# Add --account=... to sbatch if the cluster charges an allocation.

set -eu  # not -x, which would print the secrets .envrc exports

mkdir -p logs
source slurm/env.sh

DATASET=${DATASET:-in21k}
RUN_NAME=${RUN_NAME:-pretrain-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}
echo "[$(date -Is)] run $RUN_NAME ($DATASET), task ${SLURM_ARRAY_TASK_ID:-none} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

exec uv run python -u -m canvit_pytorch.pretrain \
    --dataset "$DATASET" \
    --shards-dir "$FEATURES_DIR/$DATASET/dinov3_vitb16/512/shards" \
    --images-dir "${IMAGES_DIR:-$IN21K_IMAGE_DIR}" \
    --validation-dir "$IMAGENET_VAL" \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --run-name "$RUN_NAME" \
    "$@"
