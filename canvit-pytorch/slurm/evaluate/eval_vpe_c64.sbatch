#!/bin/bash
#SBATCH --job-name=canvit-eval-vpe-c64
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/eval-vpe-c64-%j.out
#SBATCH --error=logs/eval-vpe-c64-%j.err

# The Viewpoint Encoding ablation pair (baseline and no-VPE checkpoints) on ADE20K
# at a 1024 px scene with a 64² canvas, over all 21 EG-C2F glimpses; probe
# training only evaluates the first 10.
#
# The batch matrix evaluates ablations at 512 px with a 32² canvas and released
# probes; these 64² probes are unpublished, as local directories
# (config.json + model.safetensors). Each is its training run's best-validation
# probe, at different steps (39,000 for the baseline, 35,500 for no-VPE), the
# selection rule of the released 32² ablation probes, so comparisons with them
# are not biased by it, but statements about this pair say so. The directories
# below were written from the runs' .pt files by
# scripts/specialize/materialize_probe.py @ fde90fc; a run of
# `python -m canvit_pytorch.specialize.ade20k` writes its selected probe as
# such a directory directly.
#
# The time limit is generous: a longer limit only costs queue latency, while a
# kill near the end costs the whole run.
#
#   sbatch --account=def-skrishna_gpu slurm/evaluate/eval_vpe_c64.sbatch

set -eu  # no -x: it would print the secrets .envrc exports (HF_TOKEN)
source slurm/ade20k.sh
mkdir -p logs results/vpe_c64

PROBES="$HOME/scratch/probe_materialized"
EPISODE=(--episode.policy entropy_coarse_to_fine
         --episode.num-glimpses 21
         --episode.canvas-grid-size 64
         --scene-size-px 1024
         --batch-size 8)

nvidia-smi --query-gpu=name,memory.total --format=csv
for arm in "baseline canvit/canvitb16-abl-baseline-2026-03-02" "no-vpe canvit/canvitb16-abl-no-vpe-2026-03-03"; do
    read -r ablation pretrained_repo <<< "$arm"
    echo "=== CanViT ADE20K, 64² canvas: abl-${ablation} ==="
    uv run python -m canvit_pytorch.evaluate ade20k-segmentation-canvit \
        --probe-repo "$PROBES/abl-${ablation}-c64" \
        --pretrained-repo "$pretrained_repo" \
        "${EPISODE[@]}" \
        --output "results/vpe_c64/abl-${ablation}_egc2f_s1024_c64.pt"
done
ls -l results/vpe_c64/
