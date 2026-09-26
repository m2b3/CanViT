#!/bin/bash
# No VPE (Viewpoint Encoding disabled).
set -euo pipefail
source "$(dirname "$0")/common.sh"
submit abl-no-vpe-200k --model.no-enable-vpe
