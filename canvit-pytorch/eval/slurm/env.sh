# SLURM job setup. Sources .envrc, then does SLURM-specific setup.
#
# .envrc is normally loaded by direnv, but SLURM jobs may not have direnv
# active, so we source it explicitly. Assumes the working directory is the
# repo root (SLURM default = submission dir).

source .envrc

echo "[env] Setting up SLURM environment..."

export PATH=$HOME/.local/bin:$PATH

# Fast node-local SSD for uv cache/venv inside the job.
if [ -n "${SLURM_TMPDIR:-}" ]; then
    export UV_CACHE_DIR="$SLURM_TMPDIR/.uv-cache"
    export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv"
    echo "[env] Using SLURM_TMPDIR for uv cache/venv"
else
    echo "[env] No SLURM_TMPDIR (interactive session)"
fi

uv sync
if [ -n "${UV_PROJECT_ENVIRONMENT:-}" ]; then
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"
fi

echo "[env] Done"
