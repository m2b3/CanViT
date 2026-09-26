#!/bin/bash
# Smaller canvas (D_can=256, asymmetric) — isolates capacity from projection mode.
# Compare with qkvo-dcan256 to measure projection effect at matched dim.
set -euo pipefail
source "$(dirname "$0")/common.sh"
submit abl-dcan256-200k --model.canvas-num-heads 2
