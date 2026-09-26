#!/bin/bash
# No F-IID rollout (R-IID only).
set -euo pipefail
source "$(dirname "$0")/common.sh"
submit abl-no-fiid-200k --n-full-start-branches 0
