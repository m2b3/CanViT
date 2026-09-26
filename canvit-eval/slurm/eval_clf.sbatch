#!/bin/bash
#SBATCH --job-name=canvit-eval-clf
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/eval-clf-%A_%a.out
#SBATCH --error=logs/eval-clf-%A_%a.err

# IN1k classification eval as a SLURM array. Each task runs one shard of the
# (filtered) job list via `canvit_eval.batch --shard k/N`. Unlike the seg
# wrapper this does NOT extract a dataset — IN1k val is read from IMAGENET_VAL
# (network-mounted), set in .envrc. Args after the script name pass through.
#
# Shard count N defaults to the array size; override with SHARD_COUNT for a
# single benchmark task over a finer partition, e.g.
#   sbatch --account=ACCT --array=0-0 --export=ALL,SHARD_COUNT=12 \
#     slurm/eval_clf.sbatch --tasks in1k-clf-ablations --n-runs 1
# Full n=1 first pass (12 variants, one task each):
#   sbatch --account=ACCT --array=0-11 slurm/eval_clf.sbatch \
#     --tasks in1k-clf-ablations --n-runs 1

set -eu  # NOT -x: would trace secret exports (HF_TOKEN) into logs
source slurm/env.sh
mkdir -p logs

N="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:?neither SHARD_COUNT nor SLURM_ARRAY_TASK_COUNT set}}"
echo "=== CanViT IN1k clf eval: shard ${SLURM_ARRAY_TASK_ID}/${N} ==="
echo "Job: ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

exec uv run python -m canvit_eval.batch --shard "${SLURM_ARRAY_TASK_ID}/${N}" "$@"
