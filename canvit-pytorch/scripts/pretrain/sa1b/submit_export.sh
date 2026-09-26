#!/bin/bash
# Submit export jobs for all tars that don't have a corresponding shard yet.
# Usage: bash sa1b/submit_export.sh [--dry-run] [--max-concurrent N]
set -euo pipefail

# Requires direnv (SA1B_TAR_DIR, SA1B_FEATURES_DIR).
# Do NOT source slurm/env.sh — that references $SLURM_TMPDIR which is unset on login nodes.
: "${SA1B_TAR_DIR:?SA1B_TAR_DIR not set (run from direnv-enabled dir)}"
: "${SA1B_FEATURES_DIR:?SA1B_FEATURES_DIR not set (run from direnv-enabled dir)}"

DRY_RUN=false
MAX_CONCURRENT="1"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

OUT_DIR="$SA1B_FEATURES_DIR/sa1b/dinov3_vitb16/1024/shards"

# Find tar indices that need exporting
pending=()
for tar in "$SA1B_TAR_DIR"/sa_*.tar; do
    idx=$(basename "$tar" .tar | sed 's/sa_//')
    shard="$OUT_DIR/sa_${idx}.pt"
    [[ -f "$shard" ]] || pending+=("$((10#$idx))")
done

if [[ ${#pending[@]} -eq 0 ]]; then
    echo "Nothing to do: all tars have shards."
    exit 0
fi

# Build array spec with optional concurrency limit
IFS=','; array_spec="${pending[*]}"; unset IFS
[[ -n "$MAX_CONCURRENT" ]] && array_spec="${array_spec}%${MAX_CONCURRENT}"

echo "Pending: ${#pending[@]} tars"
echo "Array:   $array_spec"

if $DRY_RUN; then
    echo "(dry run, not submitting)"
    exit 0
fi

sbatch --array="$array_spec" sa1b/export_features.sh
