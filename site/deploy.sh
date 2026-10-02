#!/usr/bin/env bash
# Publish the project page to https://m2b3.github.io/CanViT/: the page's files, the talks under slides/, the recorded
# bundles they load (record_bundles.sh) and the files of data/talk the talks name (their experiments write them) become
# the only commit of the gh-pages branch, which GitHub Pages serves. Recorded data are generated and never committed
# to main; replacing gh-pages at every deploy keeps them out of all history. A talk ships its committed files except
# its notes (Markdown), working pages (names starting with _) and the experiments that compute its data
# (experiments/).
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
bundles=$(grep -ho 'data/[a-z0-9-]*' site/index.html site/slides/*/index.html | grep -v '^data/talk$' | sort -u)
[ -n "$bundles" ] || { echo "site/index.html names no bundle under data/" >&2; exit 1; }
talks=$(git ls-files site/slides | grep -E '^site/slides/[0-9]{4}-[0-9]{2}-[0-9]{2}-[^/]+/' | grep -v -E '\.md$|/_[^/]*$|/experiments/')
# The talks' own data: the files and directories under data/talk that their pages and scripts name (a trailing period
# ends a sentence in a comment).
talk_data=$(grep -ho 'data/talk/[A-Za-z0-9_./#-]*' site/slides/*/index.html site/slides/*/*.js | sed 's/\.$//' | sort -u)
for path in $talk_data; do
  [ -e "site/$path" ] || { echo "site/$path is missing: the talk's experiments/build_deck_data.sh writes it" >&2; exit 1; }
done
[ -f site/slides/node_modules/reveal.js/dist/reveal.mjs ] || { echo "reveal.js is missing: npm ci --prefix site/slides" >&2; exit 1; }
for bundle in $bundles; do
  [ -f "site/$bundle/manifest.json" ] || { echo "site/$bundle is missing: run site/record_bundles.sh" >&2; exit 1; }
done
[ -f site/assets/paper/arch_overview.svg ] || { echo "site/assets/paper/arch_overview.svg is missing: run site/copy_paper_figures.sh" >&2; exit 1; }

out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
cp -R site/index.html site/live.html site/css site/js "$out/"
mkdir "$out/data" && for bundle in $bundles; do cp -R "site/$bundle" "$out/data/"; done
mkdir "$out/assets" && cp -R site/assets/fonts site/assets/logos site/assets/paper site/assets/authors site/assets/qr \
  site/assets/social-preview.png "$out/assets/"
mkdir -p "$out/slides/node_modules/reveal.js" && cp site/slides/deck.js site/slides/deck.css "$out/slides/"
cp -R site/slides/node_modules/reveal.js/dist "$out/slides/node_modules/reveal.js/"
for file in $talks; do mkdir -p "$out/$(dirname "${file#site/}")" && cp "$file" "$out/${file#site/}"; done
for path in $talk_data; do mkdir -p "$out/$(dirname "$path")" && cp -R "site/$path" "$out/$(dirname "$path")/"; done
find "$out" -type f | sed "s|^$out/||" | sort
du -sh "$out"

if [ "${1:-}" = "--push" ]; then
  uvx ghp-import --no-jekyll --no-history --force --push --branch gh-pages \
    --message "Project page from $(git rev-parse --short HEAD)" "$out"
fi
