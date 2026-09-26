#!/bin/bash
# No dense (spatial) supervision — CLS loss only.
# Tests whether dense distillation is what gives CanViT its spatial awareness.
set -euo pipefail
source "$(dirname "$0")/common.sh"
submit abl-no-dense-200k --no-enable-scene-patches-loss
