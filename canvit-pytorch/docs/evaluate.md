# Evaluation (`canvit_pytorch.evaluate`)

The evaluation package runs CanViT, frozen DINOv3 baselines and the latency
benchmark used by the paper. Commands below run from `canvit-pytorch/`.

## Setup

Install the evaluation dependencies:

```bash
uv sync --extra evaluate
```

Set the two dataset variables before running a dataset task:

```bash
export ADE20K_ROOT=/path/to/ADEChallengeData2016
export IMAGENET_VAL=/path/to/ILSVRC2012/val
```

`ADE20K_ROOT` must contain `images/validation` and `annotations/validation`.
The evaluation transform resizes both to the configured square scene size
without cropping. `IMAGENET_VAL` must be an ImageNet-1k `ImageFolder` tree with
one subdirectory per class; its images are resized on the short side and
center-cropped.

CanViT checkpoints and probes resolve from the [Hugging Face
organization](https://huggingface.co/canvit) by default. Set
`CANVIT_HUB_ROOT` to a local mirror whose immediate children are the repository
directories when running without those downloads. DINOv3 teacher repositories
remain gated and require an accepted license and a token.

`.envrc.example` contains these variables alongside the storage and credential
settings used by the other workflows.

## Commands

Inspect the command or a command's options before running it:

```bash
uv run python -m canvit_pytorch.evaluate --help
uv run python -m canvit_pytorch.evaluate COMMAND --help
```

| Command | Output |
| --- | --- |
| `ade20k-segmentation-canvit` | Dataset mIoU after every glimpse from a probe on CanViT's canvas. |
| `ade20k-segmentation-dinov3` | Dataset mIoU from a frozen DINOv3 patch probe after one forward pass. |
| `imagenet-classification` | Top-5 class indices after every glimpse from the recurrent CLS readout. |
| `reconstruction` | Cosine similarity between CanViT's predicted teacher features and the teacher features after every glimpse. |
| `mask-iou-dinov3` | Per-image, per-class ADE20K intersection, union and ground-truth area for DINOv3. |
| `mask-iou-canvit` | The same per-mask counts for CanViT after every glimpse and selected canvas grid. |
| `batch` | The paper's evaluation matrix, with each job in a fresh process. |
| `latency` | One configuration's forward latency and CUDA peak memory. |
| `latency-matrix` | The CPU and CUDA latency matrix. |
| `latency-summary` | Statistics and drift checks for JSONL latency records. |

CanViT episode tasks accept `--episode.policy`,
`--episode.num-glimpses`, `--episode.canvas-grid-size` and
`--episode.glimpse-size-px`. The default episode is 21 coarse-to-fine glimpses
of 128 px onto a 32×32 canvas. Reconstruction defaults to 10 random-IID
glimpses. Dataset tasks use bfloat16 autocast by default; pass `--no-amp` to
disable it. `--device` defaults to `cuda`.

The ADE20K CanViT command requires a probe trained for the selected checkpoint,
scene size and canvas grid:

```bash
uv run python -m canvit_pytorch.evaluate ade20k-segmentation-canvit \
  --probe-repo canvit/probe-ade20k-40k-s512-c64-in21k \
  --episode.policy entropy_coarse_to_fine \
  --episode.canvas-grid-size 64 \
  --batch-size 8 \
  --output results/ade20k_seg.pt
```

Evaluate the frozen and fine-tuned ImageNet classifiers separately. The
fine-tuned classifier is the default; `classifier:frozen` fuses the released
DINOv3 ViT-B/16 CLS probe into the frozen CanViT readout.

```bash
uv run python -m canvit_pytorch.evaluate imagenet-classification \
  classifier:frozen \
  --output results/in1k_frozen.pt

uv run python -m canvit_pytorch.evaluate imagenet-classification \
  classifier:finetuned \
  --output results/in1k_finetuned.pt
```

## Result files

Every dataset result is a `torch.save` file with a `metadata` entry containing
the task configuration and runtime provenance. The result-specific payloads
are:

- ADE20K segmentation: `mious`, keyed by `t0`, `t1`, and so on. DINOv3 has
  only `t0`.
- ImageNet-1k: `top_k_preds`, an `int16 [N, T, 5]` tensor, and `labels`, an
  `int16 [N]` tensor.
- Reconstruction: `per_timestep`, with `t`, `scene_cos_raw`, `cls_cos_raw`,
  `scene_cos_norm` and `cls_cos_norm`, plus `n_images`. The `norm` metrics
  compare in the teacher's standardized space; the `raw` metrics
  destandardize CanViT's predictions before comparing them with raw teacher
  features.

The mask-IoU commands write Parquet tables and a JSON sidecar with the runs
that produced them. The default paths are:

```text
results/ade20k_obj/dv3_iou.parquet
results/ade20k_obj/dv3_iou.json
results/ade20k_obj/canvit_iou.parquet
results/ade20k_obj/canvit_iou.json
```

The DINOv3 table has one row for every validation image and ADE20K class at
each requested input size. The CanViT table adds a timestep and canvas
resolution. A CanViT invocation replaces the rows for each grid it evaluates
and preserves rows for other grids already present in the output.

## Evaluation matrix

Preview the ordered jobs before allocating devices:

```bash
uv run python -m canvit_pytorch.evaluate batch --dry-run
```

Run the default matrix with five runs for each stochastic policy:

```bash
uv run python -m canvit_pytorch.evaluate batch --num-runs 5
```

The default groups are `ade20k_seg`, `in1k_clf_frozen`,
`in1k_clf_finetuned` and `recon`. Each group writes beneath
`<out-dir>/<group>/`. CanViT and ImageNet job names include the policy, scene
size, canvas grid, UTC timestamp in `YYYYMMDDTHHMMSSZ` form and, for repeated
stochastic runs, `_r<index>`. DINOv3 jobs name their variant and input size.
Deterministic policies run once. Jobs are ordered so every configuration gets
its first run before repeats.

Use `--groups`, `--policies` and `--grids` to filter the matrix;
`--include-extra-grids` adds the canvas-grid sweep. `--device` selects the
device and `--max-batch-size` caps each task's batch size. `--skip-existing`
checks for an output with the same job name under any timestamp. Use
`--shard-index` and `--shard-count` to run slices of the ordered list in
separate array tasks.

## Per-mask IoU

The default commands reproduce the DINOv3 128 px table and the CanViT 8²,
16², 32² and 64² canvas tables:

```bash
uv run python -m canvit_pytorch.evaluate mask-iou-dinov3
uv run python -m canvit_pytorch.evaluate mask-iou-canvit
```

Pass `--input-sizes-px` or `--canvas-grid-sizes` to evaluate other entries.
The CanViT policy defaults to entropy-guided coarse-to-fine with 21 glimpses;
the policy reads the segmentation probe's entropy from the canvas.

## Latency

Measure one configuration:

```bash
uv run python -m canvit_pytorch.evaluate latency \
  --device cpu \
  --scene-size-px 512
```

The result is a JSONL file under `results/latency/`. Each file starts with a
configuration row, records warmup and timed iterations as they run, and adds a
CUDA peak-memory row for CUDA measurements. The default protocol uses three
untimed warmups, at least five timed iterations, a 20-second budget and a
500-iteration cap. Each timed iteration is between device synchronizations.

Run the paper's matrix with a fresh process for each configuration:

```bash
uv run python -m canvit_pytorch.evaluate latency-matrix --dry-run
uv run python -m canvit_pytorch.evaluate latency-matrix
```

The matrix uses three passes, shuffles configuration order per pass, and
checks that the machine is idle before measuring. `--no-check-idle` disables
that precondition explicitly.

Summarize any records produced by either command:

```bash
uv run python -m canvit_pytorch.evaluate latency-summary \
  --records results/latency/*.jsonl
```

## Tests

```bash
uv run --all-extras pytest -q tests/evaluate tests/benchmarks
```
