# ImageNet-1k fine-tuning in JAX/Flax NNX

This trainer fine-tunes CanViT-B end to end for ImageNet-1k classification
with the paper's LP-FT recipe, the recipe that
`canvit_pytorch.specialize.in1k_tpu` runs on PyTorch/XLA and that produced the
released JAX fine-tuned checkpoints, `canvit_pytorch.hub.repos.FINETUNED_IN1K_NNX`.

It starts from a released pretrained CanViT-B, its CLS readout fused with the
DINOv3 ViT-B/16 ImageNet-1k probe, and trains every parameter with AdamW
(linear warmup, then cosine decay). Each scene is seen through F-IID
glimpses, with a cross-entropy loss after every glimpse and backpropagation
through all of them. `canvit_nnx.specialize.in1k.train.Recipe` holds the
recipe. The batch is split over the devices of a host, or of every host of a
multi-host slice. Computation is bfloat16 by default; on TPU, Canvas Attention
runs as the fused Pallas kernels of `canvit_nnx.model.fused_canvas_attention`.

## Data

The trainer reads ImageNet-1k TFRecord shards from a local directory (a
mounted bucket works): files named `train-*` and `validation-*` whose records
hold `image/encoded` (a JPEG) and `image/class/label` (1 to 1000), the layout
the PyTorch trainer also reads. Training scenes are random resized crops
(`CROP_SCALE` of the image, in `canvit_nnx.specialize.in1k.data`) with
horizontal flips; validation scenes are resized and center-cropped as in
evaluation. On a multi-host slice, each host reads its share of the training
shards and its rows of every validation batch.

## Run

```bash
cd canvit-nnx
uv sync --extra train
uv run python -m canvit_nnx.specialize.in1k.train --data-dir /path/to/imagenet --run-dir runs/lpft-in21k
```

`--help` lists every option. The recipe's defaults are the paper's run;
`--recipe.pretraining-dataset in1k` fine-tunes the ImageNet-1k-pretrained
CanViT-B instead. `--stop-after-step` ends an invocation early within the full
run's schedule. On a multi-host slice, every host runs the same command with
`--multi-host`, and the run directory must be one that every host sees, such
as a mounted bucket.

## Run directory

The run directory holds the recipe, `metrics.jsonl` and checkpoints, laid out
as `canvit_nnx.training.run_directory` describes. The metrics are training windows
(`train/...`, every `--log-every` steps) and validations (`val/...`, every
epoch and when an invocation ends). Validation reads every validation image
once, with the same viewpoints at every validation of a run. A checkpoint's
classifier loads with
`CanViTForImageClassification.from_pretrained("<run>/checkpoints/step_.../model")`.

Rerunning the command resumes from the newest checkpoint: the model, the
optimizer and the step continue exactly, and the training data restarts on a
new shuffle.
