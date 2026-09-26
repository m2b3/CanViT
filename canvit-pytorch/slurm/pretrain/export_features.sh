#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --array=0-89%20
#SBATCH --output=logs/export_%A_%a.out
#SBATCH --error=logs/export_%A_%a.err

# ==============================================================================
# Export DINOv3 teacher features for ImageNet-21k
# ==============================================================================
#
# RECOVERY GUARANTEES:
#   - Atomic writes: shards are .tmp until complete, then renamed to .pt
#   - Resume-safe: existing .pt files are skipped automatically
#   - Self-describing: each shard contains full metadata for verification
#   - No partial corruption: a shard either exists and is complete, or doesn't
#
# USAGE:
#   # Submit array job (over-estimate is fine, empty jobs exit quickly)
#   sbatch --array=0-99%20 slurm/export_features.sh
#
#   # If your cluster requires an allocation account:
#   sbatch --account=my_project_name --array=0-99%20 slurm/export_features.sh
#
#   # IN1k VALIDATION, for held-out evaluation of a distilled arm. 50,000
#   # images -> 13 shards -> ~79 GB, one array task. The index is the existing
#   # UNshuffled val.parquet and the destination is scratch, because this
#   # artifact's real home is GCS.
#   DATASET=in1k_val \
#   PARQUET=$INDEX_DIR/val.parquet \
#   IMAGE_ROOT=$IN1K_VAL_IMAGE_DIR \
#   OUT_DIR=$SCRATCH/dinov3_in1k_val_vitb16_512 \
#     sbatch --array=0-0 slurm/export_features.sh
#
# MONITOR:
#   squeue -u $USER                                    # job status
#   ls $OUT_DIR/shards/*.pt 2>/dev/null | wc -l        # completed shards
#   tail -f logs/export_${SLURM_JOB_ID}_*.out          # live logs
#   grep -l "Error\|Traceback" logs/export_*.out       # find failures
#
# RECOVERY:
#   Just resubmit. Existing shards are skipped. Or submit specific tasks:
#   sbatch --array=3,17,42 slurm/export_features.sh
#
# ==============================================================================

set -euo pipefail

# ==============================================================================
# SETUP - source env.sh FIRST (defines paths)
# ==============================================================================

source slurm/env.sh
mkdir -p logs

# ==============================================================================
# CONFIG
# ==============================================================================

# Experiment-specific. DATASET + IMAGE_ROOT are overridable from the
# submission environment for non-in21k exports, e.g.:
#   DATASET=in1k IMAGE_ROOT=/path/to/in1k/train sbatch --array=0-8 slurm/export_features.sh
DATASET=${DATASET:-in21k}
TEACHER_REPO_ID="facebook/dinov3-vitb16-pretrain-lvd1689m"
IMAGE_SIZE=512
SHARD_SIZE=4096
SHARDS_PER_JOB=36

# Derived from env.sh, and OVERRIDABLE — the derivation encodes two
# assumptions that hold for the IN21k and IN1k TRAIN exports and for nothing
# else:
#
#   * that the index is SHUFFLED. A training loader reads shards sequentially
#     and needs mixed classes per shard; an evaluation export is joined by
#     filename and does not care, so `val.parquet` has no `-shuffled` twin and
#     needs none.
#   * that the destination is the shared PROJECT allocation. Right for a
#     durable corpus, wrong for a transient artifact whose destination is GCS —
#     and def-skrishna sits at 26 of 33 TiB.
#
# Left as bare assignments these were not defaults but hardcodes: they do not
# fail when they become wrong, they silently point the job somewhere else.
PARQUET="${PARQUET:-$INDEX_DIR/${DATASET}-shuffled.parquet}"
IMAGE_ROOT="${IMAGE_ROOT:-$IN21K_IMAGE_DIR}"
OUT_DIR="${OUT_DIR:-$FEATURES_DIR/${DATASET}/dinov3_vitb16/${IMAGE_SIZE}}"

JOB_ID=${SLURM_ARRAY_TASK_ID:?Must run as array job}
START_SHARD=$((JOB_ID * SHARDS_PER_JOB))
END_SHARD=$((START_SHARD + SHARDS_PER_JOB))

# ==============================================================================
# LOGGING - Print everything needed to understand/reproduce this run
# ==============================================================================

echo "========================================"
echo "SLURM_JOB_ID:       $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $JOB_ID"
echo "Node:               $(hostname)"
echo "Date:               $(date -Iseconds)"
echo "Git commit:         $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "========================================"
echo "GPU:                $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "========================================"
echo "PARQUET:            $PARQUET"
echo "IMAGE_ROOT:         $IMAGE_ROOT"
echo "OUT_DIR:            $OUT_DIR"
echo "TEACHER_REPO_ID:    $TEACHER_REPO_ID"
echo "IMAGE_SIZE:         $IMAGE_SIZE"
echo "SHARD_SIZE:         $SHARD_SIZE"
echo "SHARDS_PER_JOB:     $SHARDS_PER_JOB"
echo "========================================"
echo "START_SHARD:        $START_SHARD"
echo "END_SHARD:          $END_SHARD"
echo "========================================"

# ==============================================================================
# SANITY CHECKS
# ==============================================================================

[[ -f "$PARQUET" ]] || { echo "FATAL: Parquet not found: $PARQUET" >&2; exit 1; }
[[ -d "$IMAGE_ROOT" ]] || { echo "FATAL: Image root not found: $IMAGE_ROOT" >&2; exit 1; }
echo "Sanity checks: PASSED"
echo "========================================"

# ==============================================================================
# RUN
# ==============================================================================

START_TIME=$(date +%s)

uv run python scripts/export_in21k_features.py \
    --parquet "$PARQUET" \
    --image-root "$IMAGE_ROOT" \
    --out-dir "$OUT_DIR" \
    --teacher-repo-id "$TEACHER_REPO_ID" \
    --image-size "$IMAGE_SIZE" \
    --shard-size "$SHARD_SIZE" \
    --start-shard "$START_SHARD" \
    --end-shard "$END_SHARD"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "========================================"
echo "Exit code:          $EXIT_CODE"
echo "Elapsed:            ${ELAPSED}s"
echo "End time:           $(date -Iseconds)"
echo "========================================"

exit $EXIT_CODE
