#!/bin/bash
# Submit one of the paper's pretraining ablations (canvit_pytorch.pretrain.ablations):
#   bash slurm/pretrain/ablation.sh no-reads
# The array has as many tasks as the ablations' schedule needs.
set -euo pipefail

SLUG=${1:?usage: bash slurm/pretrain/ablation.sh SLUG}
TASKS=$(uv run python -c "
import math
from canvit_pytorch.pretrain.ablations import ABLATION_TOTAL_STEPS
from canvit_pytorch.pretrain.config import PretrainingConfig
from dataclasses import fields
steps_per_job = next(f.default for f in fields(PretrainingConfig) if f.name == 'steps_per_job')
print(math.ceil(ABLATION_TOTAL_STEPS / steps_per_job))
")
RUN_NAME=ablation-$SLUG sbatch --array="0-$((TASKS - 1))%1" --job-name="canvit-ablation-$SLUG" \
    "$(dirname "$0")/train.sbatch" --ablation "$SLUG"
