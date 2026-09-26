#!/bin/bash
#SBATCH --job-name=ade20k-dinov3
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ade20k-dinov3-%j.out
#SBATCH --error=logs/ade20k-dinov3-%j.err

# DINOv3 baseline probe training at a given resolution.
# Usage:
#   sbatch slurm/train_ade20k_dinov3.sbatch --scene-size 512
#   sbatch --account=my_project_name slurm/train_ade20k_dinov3.sbatch --scene-size 512
#   sbatch slurm/train_ade20k_dinov3.sbatch --scene-size 256 --teacher-repo facebook/dinov3-vits16-pretrain-lvd1689m

set -eu  # NOT -x: would trace secret exports (HF_TOKEN, etc.) into logs
source slurm/setup.sh
mkdir -p logs

echo "=== ADE20K DINOv3 Probe Training ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"

exec uv run python -m canvit_specialize.training.ade20k train-dinov3-probe "$@"
