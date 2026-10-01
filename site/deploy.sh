#!/usr/bin/env bash
# Publish the project page to https://m2b3.github.io/CanViT/: the page's files and the recorded bundles
# (record_bundles.sh) become the only commit of the gh-pages branch, which GitHub Pages serves. Bundles are
# generated and never committed to main; replacing gh-pages at every deploy keeps them out of all history.
#
#   bash site/deploy.sh          # check and stage the page, list what would be published
#   bash site/deploy.sh --push   # publish
set -euo pipefail
site=$(cd "$(dirname "$0")" && pwd)
cd "$site/.."

python3 site/check_paper_numbers.py
[ -z "$(git status --porcelain -- site)" ] || { echo "site/ has uncommitted changes: commit them first" >&2; exit 1; }
# The bundles the pages load, as index.html and live.html name them; other recorded bundles stay local.
bundles=$(grep -ho 'data/[a-z0-9-]*' site/index.html site/live.html | sort -u)
[ -n "$bundles" ] || { echo "site/index.html and site/live.html name no bundle under data/" >&2; exit 1; }
for bundle in $bundles; do
  [ -f "site/$bundle/manifest.json" ] || { echo "site/$bundle is missing: run site/record_bundles.sh" >&2; exit 1; }
done
[ -f site/assets/paper/arch_overview.svg ] || { echo "site/assets/paper/arch_overview.svg is missing: run site/copy_paper_figures.sh" >&2; exit 1; }

out=$(mktemp -d)
cp -R site/index.html site/live.html site/css site/js "$out/"
mkdir "$out/data" && for bundle in $bundles; do cp -R "site/$bundle" "$out/data/"; done
mkdir "$out/assets" && cp -R site/assets/fonts site/assets/logos site/assets/paper site/assets/social-preview.png "$out/assets/"
find "$out" -type f | sed "s|^$out/||" | sort
du -sh "$out"

if [ "${1:-}" = "--push" ]; then
  uvx ghp-import --no-jekyll --no-history --force --push --branch gh-pages \
    --message "Project page from $(git rev-parse --short HEAD)" "$out"
fi
