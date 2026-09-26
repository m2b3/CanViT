#!/bin/bash
# No temporal BPTT: chunk_size=1 (isolated per-step backward).
# continue_prob=0.75 preserves E[n_glimpses]=4 (baseline: chunk_size=2, p=0.5).
set -euo pipefail
source "$(dirname "$0")/common.sh"
submit abl-no-bptt-200k --chunk-size 1 --continue-prob 0.75
