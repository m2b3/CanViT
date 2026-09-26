#!/bin/bash
#SBATCH --job-name=ade20k-dinov3-probe
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ade20k-dinov3-probe-%j.out
#SBATCH --error=logs/ade20k-dinov3-probe-%j.err

# An ADE20K probe on the patch features of a frozen DINOv3 ViT. Arguments go to the
# `dinov3-probe` subcommand, e.g.
#   sbatch slurm/specialize/train_ade20k_dinov3_probe.sbatch --input-size-px 512
#   sbatch slurm/specialize/train_ade20k_dinov3_probe.sbatch --variant vits16 --input-size-px 128

set -eu  # not -x, which would print secrets exported by .envrc (HF_TOKEN, COMET_API_KEY)
source slurm/ade20k.sh
mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

exec uv run python -m canvit_pytorch.specialize.ade20k dinov3-probe \
    --output-dir "${CHECKPOINTS_DIR:?CHECKPOINTS_DIR not set; check .envrc}/canvit-ade20k-probes" "$@"
