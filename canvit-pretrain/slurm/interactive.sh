#!/bin/bash
# Usage (from repo root): bash slurm/interactive.sh [time] [mem]
# If your cluster requires an allocation account, set SLURM_ACCOUNT=my_project_name.

TIME="${1:-0:20:00}"
MEM="${2:-32G}"
REPO_DIR="$(pwd)"
ACCOUNT_ARG=()
if [ -n "${SLURM_ACCOUNT:-}" ]; then
    ACCOUNT_ARG=(--account="$SLURM_ACCOUNT")
fi

exec srun "${ACCOUNT_ARG[@]}" --gres=gpu:1 --mem="$MEM" --cpus-per-task=16 --time="$TIME" --pty bash -c "
cd '$REPO_DIR'
source slurm/env.sh
uv sync
exec bash
"
