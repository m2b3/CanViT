# SLURM job setup. Sources .envrc for env vars, then does SLURM-specific stuff.
#
# .envrc is normally loaded automatically by direnv, but we source it
# explicitly here since SLURM jobs may not have direnv active.
# Assumes working directory is repo root (SLURM default = submission dir).

source .envrc

echo "[env] Setting up SLURM environment..."

export PATH=$HOME/.local/bin:$PATH
module load java/17.0.6 2>/dev/null && echo "[env] Loaded java/17.0.6" || true

# Use fast local SSD for uv cache/venv in SLURM jobs
if [ -n "$SLURM_TMPDIR" ]; then
    export UV_CACHE_DIR="$SLURM_TMPDIR/.uv-cache"
    export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv"
    echo "[env] Using SLURM_TMPDIR for uv cache/venv"
else
    echo "[env] No SLURM_TMPDIR (interactive session)"
fi

uv sync
if [ -n "$UV_PROJECT_ENVIRONMENT" ]; then
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"
else
    echo "[env] No venv activation (use 'uv run' directly)"
fi

echo "[env] Done"
