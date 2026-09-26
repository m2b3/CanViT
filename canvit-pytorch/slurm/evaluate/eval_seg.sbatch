#!/bin/bash
#SBATCH --job-name=canvit-eval-seg
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/eval-seg-%A_%a.out
#SBATCH --error=logs/eval-seg-%A_%a.err

# ADE20K evaluation matrix as a SLURM array: each task runs one shard of the
# filtered job list, with ADE20K extracted to node-local storage. Arguments after
# the script name go to `python -m canvit_pytorch.evaluate batch`.
#
# The shard count defaults to the array size; SHARD_COUNT overrides it, e.g. to
# run one shard of a finer partition:
#   sbatch --account=ACCT --array=0-0 --export=ALL,SHARD_COUNT=12 \
#     slurm/evaluate/eval_seg.sbatch --groups ade20k_seg_ablations \
#     --policies entropy_coarse_to_fine --skip-existing
# All twelve shards:
#   sbatch --account=ACCT --array=0-11 slurm/evaluate/eval_seg.sbatch \
#     --groups ade20k_seg_ablations --policies entropy_coarse_to_fine --skip-existing

set -eu  # no -x: it would print the secrets .envrc exports (HF_TOKEN)
source slurm/ade20k.sh
mkdir -p logs

N="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:?neither SHARD_COUNT nor SLURM_ARRAY_TASK_COUNT set}}"
echo "=== CanViT ADE20K evaluation: shard ${SLURM_ARRAY_TASK_ID} of ${N} ==="
echo "Job: ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

exec uv run python -m canvit_pytorch.evaluate batch --shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count "${N}" "$@"
