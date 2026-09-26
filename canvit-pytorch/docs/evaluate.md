# Evaluation (`canvit_pytorch.evaluate`)

Evaluation of [CanViT](../../README.md), the Canvas Vision Transformer, and of
its DINOv3 baselines on the paper's benchmarks, plus the latency benchmark.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/). From `canvit-pytorch/`:

```bash
uv sync --all-extras
```

The datasets are read from two environment variables when a command runs (a
missing one fails with its name; `--help` works without them):

```bash
export ADE20K_ROOT=/path/to/ADEChallengeData2016   # images/ and annotations/
export IMAGENET_VAL=/path/to/ILSVRC2012/val        # one subdirectory per class
```

`.envrc.example` sets them along with the rest of the machine-specific
environment. Checkpoints and probes load from the `canvit` organization on the
Hugging Face Hub, or from a local directory laid out the same way when
`CANVIT_HUB_ROOT` points to one.

## Commands

```bash
uv run python -m canvit_pytorch.evaluate --help             # list the commands
uv run python -m canvit_pytorch.evaluate COMMAND --help     # a command's options
```

| Command | Result |
|---|---|
| `ade20k-segmentation-canvit` | ADE20K mIoU after every glimpse, from a linear probe on the canvas |
| `ade20k-segmentation-dinov3` | ADE20K mIoU of a frozen DINOv3 with a linear probe, the passive baseline |
| `imagenet-classification` | ImageNet-1k top-5 predictions after every glimpse, frozen with a fused probe or fine-tuned |
| `reconstruction` | Cosine similarity of CanViT's predictions of its teacher's features with the teacher's, after every glimpse (the ablation metric) |
| `mask-iou-dinov3`, `mask-iou-canvit` | Per-mask ADE20K IoU (Figure 5A-B, Appendix D.5) |
| `batch` | The paper's evaluation matrix |
| `latency`, `latency-matrix`, `latency-summary` | Inference latency and peak memory (Appendix F) |

The CanViT ADE20K, ImageNet-1k and reconstruction commands take an episode:
`--episode.policy` (one of `canvit_pytorch.policies.POLICIES`),
`--episode.num-glimpses`, `--episode.canvas-grid-size` and
`--episode.glimpse-size-px`. The defaults are 21 C2F glimpses of 128 px onto a
32×32 canvas, except for reconstruction, which defaults to the ablation study's
10 R-IID glimpses. Models run under bfloat16 autocast unless `--no-amp` is
passed; `--device` defaults to `cuda`.

The flagship on ADE20K with a 64×64 canvas under EG-C2F:

```bash
uv run python -m canvit_pytorch.evaluate ade20k-segmentation-canvit \
    --probe-repo canvit/probe-ade20k-40k-s512-c64-in21k \
    --episode.policy entropy_coarse_to_fine --episode.canvas-grid-size 64 --batch-size 8 \
    --output results/ade20k_seg.pt
```

Frozen ImageNet-1k classification (`classifier:finetuned`, the default,
evaluates the fine-tuned checkpoint):

```bash
uv run python -m canvit_pytorch.evaluate imagenet-classification classifier:frozen --output results/in1k_frozen.pt
```

### Outputs

Each result records its configuration and provenance (code revision, versions,
device, command line, SLURM job) under `metadata`.

- ADE20K segmentation: a `.pt` file with `{"mious": {"t0": ..., "t1": ...}, "metadata": ...}`,
  dataset-level mIoU after each glimpse (only `t0` for DINOv3).
- ImageNet-1k: a `.pt` file with `{"top_k_preds": int16 [N, T, 5], "labels": int16 [N], "metadata": ...}`.
- Reconstruction: a `.pt` file with `{"per_timestep": [{"t", "scene_cos_raw", "cls_cos_raw",
  "scene_cos_norm", "cls_cos_norm"}, ...], "n_images", "metadata"}`. "norm" compares in the
  teacher's standardized space, as the paper reports; "raw" compares the destandardized
  predictions with the teacher's raw features.
- Mask IoU: a parquet table with one row per (image, class) and glimpse, and a JSON file of
  the same name describing the runs behind it (see below).

## The evaluation matrix

```bash
uv run python -m canvit_pytorch.evaluate batch --dry-run     # list the jobs
uv run python -m canvit_pytorch.evaluate batch --num-runs 5
```

`batch` runs every configuration behind the paper's ADE20K, ImageNet-1k and
ablation results, one job at a time, each in a fresh process. Jobs belong to
groups (`--groups`, by default `ade20k_seg in1k_clf_frozen in1k_clf_finetuned
recon`), and each group saves under `<out-dir>/<group>/` with names such as
`coarse_to_fine_s512_c32_<UTC timestamp>_r0.pt`: policy, scene size, canvas
grid, run. Stochastic policies run `--num-runs` times, deterministic ones once;
every configuration runs once before any repeats.

`--include-extra-grids` adds the canvas grid sweep, `--policies` and `--grids`
filter the jobs, `--device` and `--max-batch-size` set every job's device and
cap its batch size, and `--skip-existing` skips jobs whose output exists under
any timestamp.
`--shard-index` and `--shard-count` split the job list, e.g. across a SLURM
array: see `slurm/evaluate/eval_seg.sbatch` and `eval_clf.sbatch`.

## Per-mask IoU

The data behind Figure 5A-B (Appendix D.5): for every validation image and
class, intersection, union and ground-truth area in pixels, so IoU can be
plotted against mask area.

```bash
uv run python -m canvit_pytorch.evaluate mask-iou-dinov3   # DINOv3 ViT-B/16 at 128 px
uv run python -m canvit_pytorch.evaluate mask-iou-canvit   # canvas grids 8² to 64², 21 EG-C2F glimpses
```

They write `results/ade20k_obj/dv3_iou.parquet` and `canvit_iou.parquet`. The
CanViT table keeps the rows of the canvas grids a run does not evaluate, so
`--canvas-grid-sizes 64` recomputes one grid.

## Latency

```bash
uv run python -m canvit_pytorch.evaluate latency --device cpu --scene-size-px 512   # one configuration
uv run python -m canvit_pytorch.evaluate latency-matrix --dry-run                   # the paper's grid
uv run python -m canvit_pytorch.evaluate latency-summary --records results/latency/*.jsonl
```

Each configuration writes `results/latency/bench_<run id>.jsonl`. The protocol
(`--protocol.*`) defaults to the paper's: 3 warmup iterations, then at least 5
timed iterations and until 20 s or 500 iterations, each between device
synchronizations; `latency-matrix` measures every configuration in 3 passes and
first checks that the machine is idle.

## Tests

```bash
uv run --all-extras pytest -q tests/evaluate tests/benchmarks
```
