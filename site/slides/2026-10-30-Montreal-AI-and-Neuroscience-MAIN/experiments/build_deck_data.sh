#!/usr/bin/env bash
# Rebuild ../../data/talk, what the deck loads that this talk's experiments compute: each experiment's sweep, its export
# of the example a slide shows, and the plots. With experiment names as arguments, only those (default: all).
#
#   ADE20K_ROOT=/path/to/ADEChallengeData2016 IMAGENETTE=/path/to/imagenette2/val \
#   REAL_LABELS=/path/to/dinov3-in1k-probes/dinov3_in1k_probes/data/in1k/real.json \
#   PAPER_EXPORTER=/path/to/CanViT-paper-exporter REBUTTAL_FLOPS=/path/to/CanViT-Toward-AVFMs/rebuttal/training_flops.py \
#   bash site/slides/2026-10-30-Montreal-AI-and-Neuroscience-MAIN/experiments/build_deck_data.sh [experiment ...]
#
# Each experiment needs only its own inputs. Sweeps write to site/.experiments/; the memory sweep takes about half an
# hour on an M4 Pro, the others minutes.
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

memory() {
  ade20k
  run memory.sweep
  run memory.export --ids 'ADE_val_00000836:signboard#1,person#2,bicycle#1'
  run memory.slide --export "$data/memory/ADE_val_00000836-signboard-person-bicycle"
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
  run metacognition.calibration
}

pretraining() {
  imagenette
  run pretraining.export --imagenette "$IMAGENETTE"
}

quickstart() {
  run quickstart.export
  run quickstart.plot
}

history_examples() {
  ade20k
  imagenette
  : "${REAL_LABELS:?set REAL_LABELS to the real.json of ImageNet-ReAL}"
  run history_examples.classify_sweep --imagenette "$IMAGENETTE" --real-labels "$REAL_LABELS"
  run history_examples.segment_sweep
  run history_examples.rank
  run history_examples.export --imagenette "$IMAGENETTE" --classification ILSVRC2012_val_00037182 \
    --segmentation ADE_val_00001509
  run history_examples.plot --scene ADE_val_00001509
}

training_cost() {
  : "${PAPER_EXPORTER:?set PAPER_EXPORTER to the CanViT-paper-exporter checkout}"
  : "${REBUTTAL_FLOPS:?set REBUTTAL_FLOPS to training_flops.py of the rebuttal}"
  uv run --no-sync --project "$PAPER_EXPORTER" python -u -m experiments.training_cost.export \
    --rebuttal-flops "$REBUTTAL_FLOPS"
}

# The street of the recorded bundles under R-IID (seed 0), as site/record_bundles.sh records street-riid, with the
# canvas after each Canvas Attention Write, for the architecture slide.
architecture() {
  ade20k
  local id=ADE_val_00001780
  uv run --no-sync --project ../../../canvit-pytorch python -u -m canvit_pytorch.viz rollout \
    --scene.image "$ADE20K_ROOT/images/validation/$id.jpg" \
    --scene.annotation "$ADE20K_ROOT/annotations/validation/$id.png" \
    --scene.source "ADE20K validation set, $id" --scene.attribution "ADE20K (MIT CSAIL)" \
    --scene.license "ADE20K terms of use" --scene.out "$data/architecture/street-riid" --scene.title "A street" \
    --policy random --capture-writes
}

all=(table_corners looking_closer memory distillation foundation metacognition pretraining quickstart history_examples
     training_cost architecture)
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
