#!/bin/bash
# No canvas reads (write-only canvas).
set -euo pipefail
source "$(dirname "$0")/common.sh"
submit abl-no-reads-200k --model.no-enable-reads
