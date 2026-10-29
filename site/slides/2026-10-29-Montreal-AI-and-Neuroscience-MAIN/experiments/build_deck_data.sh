#!/usr/bin/env bash
# Rebuild ../../data/talk, what the deck loads that this talk's experiments compute: each experiment's sweep, its export
# of the example a slide shows, and the plots. With experiment names as arguments, only those (default: all).
#
#   ADE20K_ROOT=/path/to/ADEChallengeData2016 IMAGENETTE=/path/to/imagenette2/val \
#   bash site/slides/2026-10-29-Montreal-AI-and-Neuroscience-MAIN/experiments/build_deck_data.sh [experiment ...]
#
# Each experiment needs only its own inputs: ADE20K_ROOT for most, IMAGENETTE for pretraining alone. Sweeps write to
# site/.experiments/.
set -euo pipefail
cd "$(dirname "$0")/.."
data=../../data/talk
work=../../.experiments

run() { uv run --no-sync --project ../../../canvit-pytorch python -u -m "experiments.$1" "${@:2}"; }
ade20k() { : "${ADE20K_ROOT:?set ADE20K_ROOT to the ADEChallengeData2016 directory}"; export ADE20K_ROOT; }
imagenette() { : "${IMAGENETTE:?set IMAGENETTE to the val directory of Imagenette 2}"; }

table_corners() {
  ade20k
  run table_corners.sweep
  run table_corners.export --ids ADE_val_00001271:table
  local export=(--exports "$work/table_corners/exports/ADE_val_00001271-table.npz" --cmap inferno --size 640)
  run table_corners.plot "${export[@]}" --separate "$data/table" \
    --panels scene truth dinov3 prob_dinov3 a b ab prob_a prob_b prob_ab entropy_ab
  run table_corners.plot "${export[@]}" --separate "$data/table-clean" --panels scene --box none
}

vit_glimpse() {
  ade20k
  run vit_glimpse.export
  run vit_glimpse.plot
  run vit_glimpse.sweep
  run vit_glimpse.examples
}

looking_closer() {
  ade20k
  run looking_closer.sweep
  run looking_closer.export --ids ADE_val_00001715:television
  local export=(--exports "$work/looking_closer/exports/ADE_val_00001715-television.npz" --size 512)
  run looking_closer.plot "${export[@]}" --separate "$data/looking-closer" \
    --panels scene glimpse_full glimpse_zoom full_in_box prob_f prob_fz
  run looking_closer.plot "${export[@]}" --separate "$data/looking-closer-clean" --panels scene --box none
  run looking_closer.dinov3_whole_scene --separate "$data/looking-closer" --export ADE_val_00001715-television
}

distillation() {
  ade20k
  run distillation.export --ids ADE_val_00000124 --sequences riid-s124 riid-s124-n21
  run distillation.plot --exports "$work/distillation/exports/ADE_val_00000124.npz" --sequences riid-s124-n21 \
    --separate "$data/distillation"
  run distillation.training_step --image ADE_val_00000124 --sequence riid-s124-n21
}

foundation() {
  ade20k
  run foundation.export --image-id ADE_val_00001509 --short-side 1024
  run foundation.decode --image-id ADE_val_00001509 --short-sides 1024
  run foundation.plot --image-id ADE_val_00001509 --short-side 1024 --classes television fireplace armchair
}

metacognition() {
  ade20k
  run metacognition.export
}

pretraining() {
  imagenette
  run pretraining.export --imagenette "$IMAGENETTE"
  run pretraining.crops
}

# The title slide's teaser: a screen of 120 validation scenes with small glimpses (about an hour), then the scenes
# chosen by eye from it (TALK.md), recorded along their viewpoints.
title_teaser() {
  ade20k
  run title_teaser.sweep --sample 120 --scales 0.25 0.35 --step 0.25 --out "$work/title_teaser/screen.jsonl"
  run title_teaser.export --sweep "$work/title_teaser/screen.jsonl" \
    --scenes ADE_val_00000419 ADE_val_00001882 ADE_val_00000203 ADE_val_00001860
}

quickstart() {
  run quickstart.export
  run quickstart.plot
}

# #gap's segmentation example.
history_examples() {
  ade20k
  run history_examples.segment_sweep
  run history_examples.rank
  run history_examples.export --segmentation ADE_val_00001509
  run history_examples.plot --scene ADE_val_00001509
}

# The cost panels of the paper's Canvas Attention figure, from the paper's own computation, for #canvas-attention.
canvas_attention() {
  uv run --project ../../../paper/exporter python -u -m experiments.canvas_attention.export
}

# The canvas's updates within a glimpse (#canvas-updates), from the paper's generator of fig:canvas-evolution.
canvas_updates() {
  ade20k
  uv run --project ../../../paper/exporter python -u -m experiments.canvas_updates.export
}

# CanViT's glimpses on a tour of a still street, moving and zooming (#next).
glimpse_tour() {
  run glimpse_tour.export
}

# The paper's "A CanViT rollout" figure (canvit_overview) built up a glimpse at a time (#canvit-rollout): the paper's
# own prep and composition, compiled once per number of glimpses shown, from the scene alone to all of them.
canvit_rollout() {
  (cd ../../../paper/exporter && uv run python -u -m canvit_paper_exporter.diagrams.canvit_overview)
  local meta=../../../paper/exporter/diagrams/outputs/canvit_overview/Places365_IMG_9600/metadata.json
  local n
  n=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['n_glimpses'])" "$meta")
  mkdir -p "$data/canvit-rollout"
  for shown in $(seq 0 "$n"); do
    typst compile --root ../../.. --input shown="$shown" experiments/canvit_rollout/stage.typ "$data/canvit-rollout/stage-$shown.svg"
  done
}

all=(table_corners vit_glimpse looking_closer distillation foundation metacognition pretraining quickstart
     history_examples canvas_attention title_teaser canvit_rollout canvas_updates glimpse_tour)
chosen=("$@")
[[ $# -gt 0 ]] || chosen=("${all[@]}")
for experiment in "${chosen[@]}"; do
  [[ " ${all[*]} " == *" $experiment "* ]] || { echo "unknown experiment $experiment; one of: ${all[*]}" >&2; exit 1; }
done
for experiment in "${chosen[@]}"; do
  echo "== $experiment" >&2
  "$experiment"
done
echo "DONE ${chosen[*]}" >&2
