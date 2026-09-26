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
for bundle in street-path shop-path ferry-path; do
  [ -f "site/data/$bundle/manifest.json" ] || { echo "site/data/$bundle is missing: run site/record_bundles.sh" >&2; exit 1; }
done

out=$(mktemp -d)
cp -R site/index.html site/style.css site/js site/data "$out/"
mkdir "$out/assets" && cp -R site/assets/logos "$out/assets/"
find "$out" -type f | sed "s|^$out/||" | sort
du -sh "$out"

if [ "${1:-}" = "--push" ]; then
  uvx ghp-import --no-jekyll --no-history --force --push --branch gh-pages \
    --message "Project page from $(git rev-parse --short HEAD)" "$out"
fi
