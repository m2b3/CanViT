# ADE20K environment setup. Source before any ADE20K SLURM job.
# Sets ADE20K_ROOT to a node-local copy of the dataset.

source slurm/env.sh

export ADE20K_ROOT="${SLURM_TMPDIR:?SLURM_TMPDIR not set — are you in a SLURM job?}/ADEChallengeData2016"

if [ ! -d "$ADE20K_ROOT" ]; then
    echo "Extracting ADE20K to $ADE20K_ROOT ..."
    unzip -q "${ADE20K_ZIP:?ADE20K_ZIP not set — check .envrc}" -d "$SLURM_TMPDIR"
    echo "Done."
fi
