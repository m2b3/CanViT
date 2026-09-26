#!/bin/bash
# Larger RW stride (6 vs default 2) — minimal canvas interaction.
# With ViT-B (12 blocks): 1 read (block 5) + 1 write (block 11).
set -euo pipefail
source "$(dirname "$0")/common.sh"
submit abl-rw-stride6-200k --model.rw-stride 6
