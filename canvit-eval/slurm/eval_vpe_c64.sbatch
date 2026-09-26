#!/bin/bash
#SBATCH --job-name=canvit-eval-vpe-c64
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/eval-vpe-c64-%j.out
#SBATCH --error=logs/eval-vpe-c64-%j.err

# The c64 Viewpoint-Encoding ablation pair, evaluated over the FULL rollout.
#
# Why this is not a `canvit_eval.batch` task: batch's `_ablation_seg_jobs` is
# pinned to (512, 32) and resolves probes by HF repo id. This pair is scene 1024
# / canvas 64 and its probes are materialized locally, so it invokes the CLI
# directly. Two arms, one job — they share the ADE20K extraction and uv sync.
#
# 🔴 --episode.n-timesteps 21 is the point. Probe TRAINING evaluates t=0..9;
# the deliverable is the t=0..20 VPE delta series, which only this pass yields.
#
# 🔴 The two --probe-repo directories are MATERIALIZED probes, not training
# output directories: a training run writes a bare `.pt`, which
# `from_pretrained` cannot read. See CanViT-specialize scripts/materialize_probe.py.
#
# ⚠️ Both probes come from best-val checkpoints selected at DIFFERENT steps
# (39,000 baseline against 35,500 no-VPE). That is the same convention the
# published c32 ablation probes use (`push_probes._find_best_pt` globs
# `*best*miou*.pt`), so the c64-vs-c32 comparison is not biased by it — but any
# statement about this pair says so.
#
# Time limit is deliberately generous: an over-long limit costs queue latency,
# never money, while a wall-clock kill near the end costs the whole run.
#
#   sbatch --account=def-skrishna_gpu slurm/eval_vpe_c64.sbatch

set -eu  # NOT -x: would trace secret exports (HF_TOKEN) into logs
source slurm/setup.sh
mkdir -p logs results/vpe_c64

M="$HOME/scratch/probe_materialized"
COMMON=(--episode.policy entropy_coarse_to_fine
        --episode.n-timesteps 21
        --episode.canvas-grid 64
        --scene-size 1024
        --batch-size 8)

echo "=== CanViT c64 VPE eval — arm 1/2: abl-baseline ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
uv run python -m canvit_eval ade20k-seg-canvit \
    --probe-repo "$M/abl-baseline-c64" \
    --episode.model-repo canvit/canvitb16-abl-baseline-2026-03-02 \
    "${COMMON[@]}" \
    --output results/vpe_c64/abl-baseline_egc2f_s1024_c64.pt

echo "=== CanViT c64 VPE eval — arm 2/2: abl-no-vpe ==="
uv run python -m canvit_eval ade20k-seg-canvit \
    --probe-repo "$M/abl-no-vpe-c64" \
    --episode.model-repo canvit/canvitb16-abl-no-vpe-2026-03-03 \
    "${COMMON[@]}" \
    --output results/vpe_c64/abl-no-vpe_egc2f_s1024_c64.pt

echo "=== BOTH ARMS DONE ==="
ls -l results/vpe_c64/
