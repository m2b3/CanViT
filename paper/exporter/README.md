# The paper's exporter

This pipeline turns CanViT evaluation results into the paper's generated inputs ([the paper](../README.md)).
Datasets write JSON, figures write SVG, PDF and PNG files, and diagrams compose Typst sources with images from
CanViT's canvas. All generated outputs go to `../exports/`.

The model, its evaluation and FLOP counts come from this repository's `canvit-pytorch`
(`canvit_pytorch`, `canvit_pytorch.evaluate`, `canvit_pytorch.flops`).

## Running

Run these commands from `paper/exporter/`:

```bash
uv sync
uv run python -m canvit_paper_exporter.run list    # datasets, figures, diagrams
uv run python -m canvit_paper_exporter.run inputs  # the data/ files each dataset and figure reads, its exports (JSON)
uv run python -m canvit_paper_exporter.run         # registered datasets, then figures
uv run python -m canvit_paper_exporter.run NAME... # chosen datasets, figures or diagrams, in order
```

The no-argument run executes every dataset, then the figures; diagrams run by name only (`run list` prints the
names). Run a dataset before any figure that depends on its JSON. Diagrams load the released CanViT-B from the
Hugging Face Hub and require `typst` on `PATH`.

Figures render with matplotlib's Agg backend and the TeX Gyre Heros font that ultraplot bundles;
`run_figure` fails if a PDF embeds any other font. The diagrams' Typst sources name system fonts (Helvetica, Liberation
Sans); where they are missing, Typst substitutes another font with only a warning, and the diagrams differ from the
committed ones.

## Inputs

`data/` is not part of this repository. The paper's data release, `canvit-paper-data.zip`, holds every file the
default run reads, in the layout below, with a README naming the commit it was made for and a `MANIFEST.sha256`.
Unzip it anywhere, link its `data/` here, then regenerate the exports and compare them with the committed ones (from
the repository root):

```bash
ln -s /path/to/canvit-paper-data/data paper/exporter/data
cd paper && just exports && git diff --stat exports latex
```

The data release accompanies the camera-ready version of the paper.

Every dataset and figure declares the files of `data/` it reads (`paths.Inputs`) and reads them only through that
declaration; `run inputs` prints them. Evaluation results live under `data/` (not tracked), in the layout
`canvit_pytorch.evaluate` writes: each file's metadata records the configuration of its run (model, probe or classifier,
policy, scene size, canvas grid), and the exporter selects and groups results by that record, never by file name. The
paper's results, except those of the learned policies, were written by an earlier evaluator and converted to this
layout, results unchanged, each file keeping its source's name, SHA-256 and metadata under `metadata.converted_from`.
The commands in the last column run from `canvit-pytorch/` and, unless noted, place new results where this exporter
reads them:

| Dataset | `data/` directory | Producer from `canvit-pytorch/` |
|---|---|---|
| `ade20k_seg` | `ade20k_seg/` | `uv run python -m canvit_pytorch.evaluate batch --out-dir ../paper/exporter/data --groups ade20k_seg --include-extra-grids` |
| `ade20k_iou_vs_obj` | `ade20k_obj/` (`dv3_iou.parquet`, `canvit_iou.parquet`) | `uv run python -m canvit_pytorch.evaluate mask-iou-dinov3 --output ../paper/exporter/data/ade20k_obj/dv3_iou.parquet`; `uv run python -m canvit_pytorch.evaluate mask-iou-canvit --output ../paper/exporter/data/ade20k_obj/canvit_iou.parquet` |
| `in1k_clf_frozen` | `in1k_clf_frozen/` | `uv run python -m canvit_pytorch.evaluate batch --out-dir ../paper/exporter/data --groups in1k_clf_frozen --include-extra-grids` |
| `in1k_clf_finetuned` | `in1k_clf_finetuned/` | `uv run python -m canvit_pytorch.evaluate batch --out-dir ../paper/exporter/data --groups in1k_clf_finetuned` |
| `ablation_recon` | `recon/` | `uv run python -m canvit_pytorch.evaluate batch --out-dir ../paper/exporter/data --groups recon` |
| `ablation_seg` | `ade20k_seg_ablations/` | `uv run python -m canvit_pytorch.evaluate batch --out-dir ../paper/exporter/data --groups ade20k_seg_ablations` |
| `ablation_in1k_clf` | `in1k_clf_ablations/` | `uv run python -m canvit_pytorch.evaluate batch --out-dir ../paper/exporter/data --groups in1k_clf_ablations` |
| `hw_bench` | `latency_bench/` | `uv run python -m canvit_pytorch.evaluate latency-matrix --output-dir ../paper/exporter/data/latency_bench` |
| `ablation_variants`, `ablation_loss` | `comet/` | pretraining histories exported from Comet, one JSON per ablation, `{metric: [{step, value, timestamp}]}` with each point's logging timestamp |
| `ablation_vpe_seg` | `ade20k_seg_ablations/`, `ade20k_seg_vpe_c64/` | as `ablation_seg`; `ade20k_seg_vpe_c64/` holds the EG-C2F results of the baseline and no-VPE ablations at scene 1024, canvas 64, with the probes fitted on that canvas (`canvit/probe-ade20k-40k-s1024-c64-abl-<ablation>`), from `slurm/evaluate/eval_vpe_c64.sbatch`, which writes them to `results/vpe_c64/` |
| `dataset_scale` | `ade20k_seg_pretrain/`, `in1k_clf_pretrain/` | `uv run python -m canvit_pytorch.evaluate batch --out-dir ../paper/exporter/data --groups ade20k_seg_pretrain`, then `--groups in1k_clf_pretrain` |
| `dataset_scale` | `in1k_clf_finetuned_pretrain/` | ImageNet-1k results of the two `FINETUNED_IN1K_NNX` checkpoints (`canvit_pytorch.evaluate imagenet-classification` with `classifier:finetuned`) |
| `architecture_control` | `architecture_control/` | probe records, cost model and pretraining chip-hours of the JAX/Flax NNX implementation, whose training code is not public; the cost model's AdaGlimpse FLOPs and parameters are those of `ARCHITECTURE_CONTROL` in [`baselines/adaglimpse-nnx`](../../baselines/adaglimpse-nnx/) (`adaglimpse_nnx.flops.segmentation_flops`) |
| `coco_detection` | `coco_detection/` | evaluation and training records of the COCO detection runs, made with the JAX/Flax NNX code that [`canvit_nnx.detect`](../../canvit-nnx/canvit_nnx/detect/) was ported from |
| `in1k_efficiency` | `in1k_clf_frozen/`, `in1k_clf_finetuned/`, `in1k_clf_finetuned_c8/` | as `in1k_clf_frozen` and `in1k_clf_finetuned`; `in1k_clf_finetuned_c8/` holds canvas-8 ImageNet-1k results of the flagship fine-tuned at canvas 8 by the JAX/Flax NNX trainer, scored from a PyTorch export of its weights (the Hub repo's current files are in the NNX format) |
| `in1k_efficiency` | `in1k_baselines/` | records of the released AdaptiveNN and GFNet checkpoints evaluated with early exit disabled: top-1, FLOPs counted over their inference loops, parameters per glimpse budget; the commands of [`baselines/gfnet-adaptivenn`](../../baselines/gfnet-adaptivenn/) write them |
| `learned_policies` | `ade20k_seg_learned_policies/`, `ade20k_seg/` | `uv run python -m canvit_pytorch.evaluate ade20k-segmentation-learned-policy` ([`docs/rl.md`](../../canvit-pytorch/docs/rl.md), "Evaluate") for each published scorer, which its run kept by validation, and for each run's last training step, given as a scorer directory; `checkpoints.json` records the scorer each result evaluated, with its training run, step and manifest. C2F and EG-C2F come from `ade20k_seg/` |
| `compute_reporting` | `compute_reporting/` | compute used after submission, by hardware: SLURM H100-equivalent hours, Cloud TPU chip-hours and list price, RTX 4090 GPU-hours recorded by the runs themselves, derived from cluster accounting, cloud billing and run records, which are not released |

`in1k_finetune_config`, `dinov3_probe_data` and `dataset_scale` read the released checkpoints' `config.json`
from the Hub, or from the directory `$CANVIT_HUB_ROOT` names. `flops` is analytic.

A figure plots the exports of the datasets it declares; `run inputs` lists them.

## Citation

See the [project citation](../../README.md#citation).

## License

MIT. See [LICENSE](../../LICENSE).
