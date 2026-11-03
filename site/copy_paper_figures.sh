#!/usr/bin/env bash
# Copy the paper's diagram that the page shows from an exports directory, by default paper/exports/, where the paper's
# exporter writes them (`just exports` in paper/). They are generated, never committed here, and published by
# deploy.sh. A talk keeps its own copies of the diagrams it shows (slides/TALK/assets/paper-diagrams/).
#
#   bash site/copy_paper_figures.sh [EXPORTS_DIR]
set -euo pipefail
site=$(cd "$(dirname "$0")" && pwd)
exports=${1:-"$site/../paper/exports"}
for figure in arch_overview.svg; do
  cp "$exports/$figure" "$site/assets/paper/$figure"
  echo "copied $figure from $exports ($(wc -c < "$site/assets/paper/$figure") bytes)"
done
