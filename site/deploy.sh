#!/usr/bin/env bash
# Publish the project page to https://m2b3.github.io/CanViT/: the page's files, the talks under slides/ and the
# recorded data they load (record_bundles.sh, data/talk) become the only commit of the gh-pages branch, which GitHub
# Pages serves. Recorded data are generated and never committed to main; replacing gh-pages at every deploy keeps them
# out of all history. A talk ships its committed files except its notes (Markdown) and working pages (names starting
# with _).
#
#   bash site/deploy.sh          # check and stage the page, list what would be published
#   bash site/deploy.sh --push   # publish
set -euo pipefail
site=$(cd "$(dirname "$0")" && pwd)
cd "$site/.."

python3 site/check_paper_numbers.py
[ -z "$(git status --porcelain -- site)" ] || { echo "site/ has uncommitted changes: commit them first" >&2; exit 1; }
# The bundles the pages and talks load, as they name them; other recorded bundles stay local. data/talk is the talks'
# own data, not a bundle.
bundles=$(grep -ho 'data/[a-z0-9-]*' site/index.html site/live.html site/slides/*/index.html | grep -v '^data/talk$' | sort -u)
[ -n "$bundles" ] || { echo "site/index.html and site/live.html name no bundle under data/" >&2; exit 1; }
talks=$(git ls-files site/slides | grep -E '^site/slides/[0-9]{4}-[0-9]{2}-[0-9]{2}-[^/]+/' | grep -v -E '\.md$|/_[^/]*$')
if grep -q 'data/talk' site/slides/*/index.html; then
  [ -d site/data/talk ] || { echo "site/data/talk is missing: the talks' throwaway exports write it" >&2; exit 1; }
fi
[ -f site/slides/node_modules/reveal.js/dist/reveal.mjs ] || { echo "reveal.js is missing: npm ci --prefix site/slides" >&2; exit 1; }
for bundle in $bundles; do
  [ -f "site/$bundle/manifest.json" ] || { echo "site/$bundle is missing: run site/record_bundles.sh" >&2; exit 1; }
done
[ -f site/assets/paper/arch_overview.svg ] || { echo "site/assets/paper/arch_overview.svg is missing: run site/copy_paper_figures.sh" >&2; exit 1; }

out=$(mktemp -d)
cp -R site/index.html site/live.html site/css site/js "$out/"
mkdir "$out/data" && for bundle in $bundles; do cp -R "site/$bundle" "$out/data/"; done
mkdir "$out/assets" && cp -R site/assets/fonts site/assets/logos site/assets/paper site/assets/authors site/assets/qr \
  site/assets/social-preview.png "$out/assets/"
mkdir -p "$out/slides/node_modules/reveal.js" && cp site/slides/deck.js site/slides/deck.css "$out/slides/"
cp -R site/slides/node_modules/reveal.js/dist "$out/slides/node_modules/reveal.js/"
for file in $talks; do mkdir -p "$out/$(dirname "${file#site/}")" && cp "$file" "$out/${file#site/}"; done
[ ! -d site/data/talk ] || cp -R site/data/talk "$out/data/"
find "$out" -type f | sed "s|^$out/||" | sort
du -sh "$out"

if [ "${1:-}" = "--push" ]; then
  uvx ghp-import --no-jekyll --no-history --force --push --branch gh-pages \
    --message "Project page from $(git rev-parse --short HEAD)" "$out"
fi
