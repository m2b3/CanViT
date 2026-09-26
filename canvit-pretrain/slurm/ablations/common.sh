# Shared ablation config. Sourced by individual scripts.
ARRAY="0-42%1"         # 43 jobs × 4992 steps/job = 214,656 steps (≥ EVAL_STEP=214,640)
WARMUP=20000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

submit() {
    local name="$1"; shift
    sbatch --array="$ARRAY" -J "$name" "$SCRIPT_DIR/train.sbatch" \
        --warmup-steps "$WARMUP" --run-name "$name" "$@"
}
