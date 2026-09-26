#!/bin/bash
# +QKVO, D_can=256 (n_h=2, d_h=128). FLOP-matched to baseline.
set -euo pipefail
source "$(dirname "$0")/common.sh"
submit abl-qkvo-dcan256-200k \
    --model.canvas-proj-mode full --model.canvas-num-heads 2
