#!/usr/bin/env bash
# Record every bundle the site shows, with the released CanViT-B and its ADE20K probe (64² canvas),
# on ADE20K validation scenes. Run from a clean checkout so each manifest's provenance names a commit:
#
#   ADE20K_ROOT=/path/to/ADEChallengeData2016 bash site/record_bundles.sh
set -euo pipefail
: "${ADE20K_ROOT:?set ADE20K_ROOT to the ADEChallengeData2016 directory}"
site=$(cd "$(dirname "$0")" && pwd)
cd "$site/../canvit-pytorch"

viz() { uv run --extra viz python -u -m canvit_pytorch.viz "$@"; }

scene() {  # ADE20K validation image id, bundle name, title -> the shared --scene.* arguments in $args
  local id=$1 name=$2 title=$3
  args=(--scene.image "$ADE20K_ROOT/images/validation/$id.jpg"
        --scene.annotation "$ADE20K_ROOT/annotations/validation/$id.png"
        --scene.source "ADE20K validation set, $id"
        --scene.attribution "ADE20K (MIT CSAIL)"
        --scene.license "ADE20K terms of use"
        --scene.out "$site/data/$name"
        --scene.title "$title")
}

# Smooth paths: keypoints are (row, col, scale) triples; each path returns to its first keypoint.
scene ADE_val_00001780 street-path "A street"
viz path "${args[@]}" --keypoints -0.2 -0.65 0.3  0.28 -0.72 0.25  0.35 -0.15 0.3  0.1 0.5 0.2  0.3 0.8 0.18  -0.55 0.75 0.25  0 0 1

scene ADE_val_00000385 shop-path "A shop"
viz path "${args[@]}" --keypoints -0.35 0.2 0.25  -0.7 0.25 0.2  -0.3 0.75 0.2  0.3 0.7 0.3  0 0 1  0.0 -0.35 0.3  0.0 -0.75 0.25

scene ADE_val_00000348 ferry-path "A ferry"
viz path "${args[@]}" --keypoints -0.05 -0.35 0.35  -0.3 0.15 0.3  -0.55 0.2 0.2  0.3 0.6 0.35  -0.3 -0.75 0.22  0 0 1

# R-IID rollouts, 21 glimpses: random viewpoints from the first glimpse on, as in pretraining (seed 0).
scene ADE_val_00001780 street-riid "A street"
viz rollout "${args[@]}" --policy random

scene ADE_val_00000385 shop-riid "A shop"
viz rollout "${args[@]}" --policy random

scene ADE_val_00000348 ferry-riid "A ferry"
viz rollout "${args[@]}" --policy random

# An EG-C2F rollout, 21 glimpses.
scene ADE_val_00001780 street-egc2f "A street, EG-C2F"
viz rollout "${args[@]}"
