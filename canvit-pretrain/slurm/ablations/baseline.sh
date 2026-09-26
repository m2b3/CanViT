#!/bin/bash
# Baseline: asymmetric (identity on canvas side), D_can=1024, VPE, additive.
set -euo pipefail
source "$(dirname "$0")/common.sh"
submit abl-baseline-200k
