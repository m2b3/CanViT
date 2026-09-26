#!/bin/bash
# TPU VM environment for ImageNet-1k fine-tuning (canvit_pytorch.specialize.in1k_tpu).
# Run by the SkyPilot YAML's setup: block; also safe to run by hand over SSH.
#
# What it does (idempotent):
#   1. Install uv (if missing) and Python 3.12.
#   2. Configure ldconfig for torch_xla (uv's python-build-standalone libpython).
#   3. Install gcsfuse (apt) if missing.
#   4. Restore uv cache from GCS (if present) to speed up the first `uv sync`.
#   5. `uv sync --project tpu`, then canvit-pytorch without its dependencies.
#
# Working directory: canvit-pytorch/ (the directory holding tpu/pyproject.toml).
# Environment: TPU VM, Ubuntu 22.04.
set -euo pipefail

SETUP_START=$(date +%s%3N)
ts() { echo "$(($(date +%s%3N) - SETUP_START))ms | $*"; }

# 1) uv + Python ─────────────────────────────────────────────────────────
ts "uv: checking..."
if ! command -v uv &>/dev/null && [ ! -f "$HOME/.local/bin/uv" ]; then
    ts "uv: installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ts "uv: installed"
else
    ts "uv: already installed"
fi
export PATH="$HOME/.local/bin:$PATH"

ts "python: installing 3.12..."
uv python install 3.12
ts "python: done"

# 2) ldconfig for torch_xla ─────────────────────────────────────────────
# uv's python-build-standalone dynamically links libpython but doesn't put it on
# the linker search path. torch_xla's _XLAC.so needs it at import time.
# See: https://github.com/astral-sh/uv/issues/6812 (as-designed upstream).
UV_PYLIB="$HOME/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/lib"
if [ -d "$UV_PYLIB" ] && ! ldconfig -p | grep -q libpython3.12.so; then
    ts "ldconfig: configuring..."
    echo "$UV_PYLIB" | sudo tee /etc/ld.so.conf.d/uv_python.conf >/dev/null
    sudo ldconfig
    ts "ldconfig: done"
fi

# 3) gcsfuse (for mounting GCS buckets) ─────────────────────────────────
# `apt-get update` is a cold-start bottleneck (18–210 s depending on zone).
if ! command -v gcsfuse &>/dev/null; then
    ts "gcsfuse: installing (apt-get update is slow)..."
    GCSFUSE_REPO="gcsfuse-$(lsb_release -c -s)"
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list >/dev/null
    curl -sSf https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc >/dev/null
    sudo apt-get update -qq && sudo apt-get install -y -qq gcsfuse
    ts "gcsfuse: installed"
else
    ts "gcsfuse: already installed"
fi

# 4) Restore uv cache from GCS (read-only; manually update via:
#    tar czf /tmp/uv-cache.tar.gz -C ~/.cache uv && \
#    gcloud storage cp /tmp/uv-cache.tar.gz "gs://${GCS_BUCKET_PREFIX}-us-central1/cache/"
# ) ─────────────────────────────────────────────────────────────────────
ZONE=$(curl -sSf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | rev | cut -d/ -f1 | rev || echo "")
REGION=${ZONE%-*}
GCS_BUCKET="gs://${GCS_BUCKET_PREFIX:?GCS_BUCKET_PREFIX not set — your per-region GCS bucket prefix, e.g. mylab}-${REGION}"
UV_CACHE_ARCHIVE="${GCS_BUCKET}/cache/uv-cache.tar.gz"
# Cache restoration is best-effort: a missing archive is expected on first run.
# We check existence separately from the download so a real auth/network failure
# during download doesn't get swallowed.
if [ -n "$REGION" ] && [ ! -d "$HOME/.cache/uv/wheels-v4" ] && gcloud storage ls "$UV_CACHE_ARCHIVE"; then
    ts "uv-cache: downloading from GCS..."
    gcloud storage cp "$UV_CACHE_ARCHIVE" /tmp/uv-cache.tar.gz
    CACHE_SIZE_MB=$(du -m /tmp/uv-cache.tar.gz | cut -f1)
    ts "uv-cache: downloaded (${CACHE_SIZE_MB}MB compressed)"
    mkdir -p ~/.cache
    tar xzf /tmp/uv-cache.tar.gz -C ~/.cache/
    rm -f /tmp/uv-cache.tar.gz
    CACHE_DIR_SIZE=$(du -sh ~/.cache/uv | cut -f1)
    ts "uv-cache: restored (${CACHE_DIR_SIZE} on disk)"
else
    ts "uv-cache: skipped (already present or no archive accessible)"
fi

# 5) Resolve training deps from tpu/pyproject.toml: torch==2.9.0 and
# torchvision==0.24.0 from https://download.pytorch.org/whl/cpu (no nvidia_*
# CUDA packages on a TPU VM) to match torch_xla==2.9.0's _XLAC.so ABI. Then
# canvit-pytorch without its dependencies (see tpu/pyproject.toml for why).
ts "uv sync --project tpu..."
uv sync --project tpu
uv pip install --python tpu/.venv --no-deps -e .
ts "uv sync: done"

ts "Setup complete: uv $(uv --version), Python $(uv run --python 3.12 -- python --version 2>&1 | tail -1), gcsfuse $(gcsfuse --version 2>/dev/null | head -1 || echo 'N/A')"
