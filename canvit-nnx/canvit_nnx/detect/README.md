# COCO object detection in JAX/Flax NNX

`canvit_nnx.detect` trains CanViT-B with a CenterNet head ("Objects as
Points", arXiv:1904.07850) on the canvas for COCO 2017 object detection, end
to end, and scores COCO average precision after each glimpse. It is the
detector of the paper's COCO experiments (`supp:coco-detection`); the defaults of
`canvit_nnx.detect.train.Recipe` are its run on the released ImageNet-21k
CanViT-B.

| Module | Holds |
|---|---|
| `targets` | CenterNet's heatmap, size and offset targets, rendered from boxes on the host |
| `model` | `CenterNetHead` and `CanViTForObjectDetection`, which reads the canvas after every glimpse |
| `loss` | CenterNet's penalty-reduced focal loss and masked L1 losses |
| `decode` | Heatmap peaks to boxes, without non-maximum suppression |
| `data` | COCO's annotations and images as scenes and targets, split over hosts |
| `photometric` | DINOv3's photometric distortion |
| `scoring` | COCO average precision with pycocotools, and bootstrap intervals |
| `evaluate` | Average precision after each glimpse under C2F viewpoints; a command line |
| `visualize` | A rollout's viewpoints, true boxes and predicted boxes, glimpse by glimpse |
| `train` | The recipe and the training command |

## Data

The data directory has COCO 2017's own layout, as distributed by
cocodataset.org: `annotations/instances_train2017.json`,
`annotations/instances_val2017.json`, `train2017/` and `val2017/`. Scoring
reads the original validation annotations, crowd regions included.

## Train

```bash
cd canvit-nnx
uv sync --extra train
uv run python -m canvit_nnx.detect.train --data-dir /path/to/coco --run-dir runs/coco
```

`--help` lists every option and its default. On a multi-host slice, every host
runs the same command with `--multi-host`, and the run directory must be one
that every host sees, such as a mounted bucket. A short run that exercises
training, validation, images and checkpoints, then resumes from its checkpoint:

```bash
smoke=(--data-dir /path/to/coco --run-dir runs/smoke --log-every 5 --validate-every 10
  --checkpoint-every 10 --validation-images 512 --validation-repeats 1)
uv run python -m canvit_nnx.detect.train "${smoke[@]}" --stop-after-step 10
uv run python -m canvit_nnx.detect.train "${smoke[@]}" --stop-after-step 20
```

## Evaluate

```bash
uv run python -m canvit_nnx.detect.evaluate --detector <Hub repository or directory> --data-dir /path/to/coco
```

prints the average precision statistics after each glimpse, means over C2F
repeats with their 95% bootstrap intervals, as JSON.

## Run directory

The run directory holds the recipe, `metrics.jsonl`, images and checkpoints,
laid out as `canvit_nnx.training.run_directory` describes. The metrics are training
windows (`train/...`: the loss and its terms, the gradient norm, the learning
rates, throughput and hardware FLOPs utilization) and validations
(`val/<statistic>_t<glimpse>`). At each validation, `images/` receives the
rollouts of the first validation scenes. A checkpoint's detector loads with
`CanViTForObjectDetection.from_pretrained("<run>/checkpoints/step_.../model")`.
Rerunning the command resumes from the newest checkpoint, with the model, the
optimizer, the viewpoints and the training data continuing exactly.
