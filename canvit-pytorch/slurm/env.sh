# SLURM job setup, sourced by every job script from the canvit-pytorch/
# directory (SLURM's default working directory is the submission directory).
#
# .envrc is normally loaded by direnv, but SLURM jobs may not have direnv
# active, so it is sourced explicitly.

source .envrc

echo "[env] Setting up SLURM environment..."

export PATH=$HOME/.local/bin:$PATH
module load java/17.0.6 2>/dev/null && echo "[env] Loaded java/17.0.6" || true

# Fast node-local SSD for the uv cache and environment inside a job.
if [ -n "${SLURM_TMPDIR:-}" ]; then
    export UV_CACHE_DIR="$SLURM_TMPDIR/.uv-cache"
    export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv"
    echo "[env] Using SLURM_TMPDIR for uv cache/venv"
else
    echo "[env] No SLURM_TMPDIR (interactive session)"
fi

uv sync --all-extras
if [ -n "${UV_PROJECT_ENVIRONMENT:-}" ]; then
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"
fi

echo "[env] Done"
