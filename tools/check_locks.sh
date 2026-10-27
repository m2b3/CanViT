#!/usr/bin/env bash
# Every uv project's lock agrees with its declarations. A lock also records the requirements of the packages it depends
# on by path, so a change to one package's pyproject can leave another project's lock stale.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
stale=0
for lock in $(git ls-files '*uv.lock'); do
  dir=$(dirname "$lock")
  (cd "$dir" && uv lock --check --quiet 2>/dev/null) || { echo "stale lock: $lock (run uv lock in $dir)" >&2; stale=1; }
done
exit $stale
