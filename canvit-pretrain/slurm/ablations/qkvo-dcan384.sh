#!/bin/bash
# +QKVO, D_can=384 (n_h=3, d_h=128). Slightly above FLOP-matched.
set -euo pipefail
source "$(dirname "$0")/common.sh"
submit abl-qkvo-dcan384-200k \
    --model.canvas-proj-mode full --model.canvas-num-heads 3
