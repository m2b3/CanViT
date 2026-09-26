#!/bin/bash
# 2× R-IID rollouts, no F-IID — controls for branch count vs. no-fiid.sh (1× R-IID).
set -euo pipefail
source "$(dirname "$0")/common.sh"
submit abl-2riid-no-fiid-200k --n-full-start-branches 0 --n-random-start-branches 2
