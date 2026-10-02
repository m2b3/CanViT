#!/usr/bin/env bash
# Copy the paper's figures that the page and the talks show from the paper's figure exports (PAPER_EXPORTS, a directory
# holding arch_overview.svg). They are generated there, never committed here, and published by deploy.sh.
#
#   PAPER_EXPORTS=/path/to/paper_exports bash site/copy_paper_figures.sh
set -euo pipefail
: "${PAPER_EXPORTS:?set PAPER_EXPORTS to the directory of the figure exports of the paper}"
site=$(cd "$(dirname "$0")" && pwd)
for figure in arch_overview.svg canvas_attention_combined.svg ade20k_seg.json in1k_clf_frozen.json; do
  cp "$PAPER_EXPORTS/$figure" "$site/assets/paper/$figure"
  echo "copied $figure ($(wc -c < "$site/assets/paper/$figure") bytes)"
done
