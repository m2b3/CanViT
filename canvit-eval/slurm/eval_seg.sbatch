#!/bin/bash
#SBATCH --job-name=canvit-eval-seg
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/eval-seg-%A_%a.out
#SBATCH --error=logs/eval-seg-%A_%a.err

# ADE20K segmentation eval as a SLURM array. Each task runs one shard of the
# (filtered) job list via `canvit_eval.batch --shard k/N`. Args after the
# script name pass through to the batch runner.
#
# Shard count N defaults to the array size (SLURM_ARRAY_TASK_COUNT); override
# with SHARD_COUNT to run a single benchmark task over a finer partition, e.g.
#   sbatch --account=ACCT --array=0-0 --export=ALL,SHARD_COUNT=12 \
#     slurm/eval_seg.sbatch --tasks ade20k-seg-ablations \
#     --policies entropy_coarse_to_fine --skip-existing
# Full 12-way run:
#   sbatch --account=ACCT --array=0-11 slurm/eval_seg.sbatch \
#     --tasks ade20k-seg-ablations --policies entropy_coarse_to_fine --skip-existing

set -eu  # NOT -x: would trace secret exports (HF_TOKEN) into logs
source slurm/setup.sh
mkdir -p logs

N="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:?neither SHARD_COUNT nor SLURM_ARRAY_TASK_COUNT set}}"
echo "=== CanViT ADE20K seg eval: shard ${SLURM_ARRAY_TASK_ID}/${N} ==="
echo "Job: ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

exec uv run python -m canvit_eval.batch --shard "${SLURM_ARRAY_TASK_ID}/${N}" "$@"
