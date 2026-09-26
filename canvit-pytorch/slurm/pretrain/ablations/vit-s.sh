#!/bin/bash
# ViT-S student backbone (D_bb=384, 6 heads, 12 blocks) instead of ViT-B.
set -euo pipefail
source "$(dirname "$0")/common.sh"
submit abl-vit-s-200k --backbone-name vits16
