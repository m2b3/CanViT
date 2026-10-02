#!/usr/bin/env bash
# Copy the paper's diagrams that the page and the talks show from paper/exports/, where the paper's exporter writes
# them (`just exports` in paper/). They are generated, never committed here, and published by deploy.sh.
#
#   bash site/copy_paper_figures.sh
set -euo pipefail
site=$(cd "$(dirname "$0")" && pwd)
for figure in arch_overview.svg canvas_attention_combined.svg; do
  cp "$site/../paper/exports/$figure" "$site/assets/paper/$figure"
  echo "copied $figure ($(wc -c < "$site/assets/paper/$figure") bytes)"
done
