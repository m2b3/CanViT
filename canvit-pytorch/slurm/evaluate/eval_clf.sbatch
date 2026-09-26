#!/bin/bash
#SBATCH --job-name=canvit-eval-clf
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/eval-clf-%A_%a.out
#SBATCH --error=logs/eval-clf-%A_%a.err

# ImageNet-1k evaluation matrix as a SLURM array: each task runs one shard of the
# filtered job list. The validation images are read from $IMAGENET_VAL, set in
# .envrc. Arguments after the script name go to `python -m canvit_pytorch.evaluate batch`.
#
# The shard count defaults to the array size; SHARD_COUNT overrides it, e.g. to
# run one shard of a finer partition:
#   sbatch --account=ACCT --array=0-0 --export=ALL,SHARD_COUNT=12 \
#     slurm/evaluate/eval_clf.sbatch --groups in1k_clf_ablations --num-runs 1
# One run of every ablation, one shard each:
#   sbatch --account=ACCT --array=0-11 slurm/evaluate/eval_clf.sbatch \
#     --groups in1k_clf_ablations --num-runs 1

set -eu  # no -x: it would print the secrets .envrc exports (HF_TOKEN)
source slurm/env.sh
mkdir -p logs

N="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:?neither SHARD_COUNT nor SLURM_ARRAY_TASK_COUNT set}}"
echo "=== CanViT ImageNet-1k evaluation: shard ${SLURM_ARRAY_TASK_ID} of ${N} ==="
echo "Job: ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

exec uv run python -m canvit_pytorch.evaluate batch --shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count "${N}" "$@"
