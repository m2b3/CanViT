# The paper's exporter

The pipeline that turns CanViT's evaluation results into the paper's tables, figures and diagrams
([the paper](../README.md)): the JSON data the manuscript's tables and prose read, matplotlib figures
as SVG and PDF, and diagrams composed with Typst from CanViT's canvas. It writes them to `../exports/`.

The model, its evaluation and FLOP counts come from this repository's `canvit-pytorch`
(`canvit_pytorch`, `canvit_pytorch.evaluate`, `canvit_pytorch.flops`).

## Running

```bash
uv sync
uv run python -m canvit_paper_exporter.run list    # datasets, figures, diagrams
uv run python -m canvit_paper_exporter.run         # the paper's datasets, then its figures
uv run python -m canvit_paper_exporter.run NAME…   # chosen datasets, figures or diagrams, in order
```

Figures read the JSON their datasets write to `../exports/`, so run datasets first: `main_results`
needs `ade20k_seg`, `in1k_clf_frozen` and `in1k_clf_finetuned`; `hw_latency` needs `hw_bench`;
`ablation_loss` needs `ablation_variants`. Diagrams run only by name; they load the released
CanViT-B from the Hugging Face Hub and need `typst` on the PATH. So do `ablation_seg` and
`ablation_in1k_clf`, downstream evaluations of the ablation checkpoints that the paper does not report.

Figures render with matplotlib's Agg backend and the TeX Gyre Heros font that ultraplot bundles;
`run_figure` fails if a PDF embeds any other font.

## Inputs

Evaluation results live under `data/` (not tracked), one directory per dataset, as
`python -m canvit_pytorch.evaluate` writes them:

| Dataset | `data/` directory | Producer |
|---|---|---|
| `ade20k_seg` | `ade20k_seg/` | `batch --groups ade20k_seg --include-extra-grids` |
| `ade20k_iou_vs_obj` | `ade20k_obj/` (`dv3_iou.parquet`, `canvit_iou.parquet`) | `mask-iou-dinov3`, `mask-iou-canvit` |
| `in1k_clf_frozen` | `in1k_frozen/` | `batch --groups in1k_clf_frozen --include-extra-grids` |
| `in1k_clf_finetuned` | `in1k_finetuned/` | `batch --groups in1k_clf_finetuned` |
| `ablation_recon` | `ablation_recon/` | `batch --groups recon` |
| `ablation_seg` | `ablation_seg/` | `batch --groups ade20k_seg_ablations` |
| `ablation_in1k_clf` | `ablation_in1k_clf/` | `batch --groups in1k_clf_ablations` |
| `hw_bench` | `latency_bench/` | `latency-matrix` |
| `ablation_variants`, `ablation_loss` | `comet/` | pretraining curves exported from Comet, one JSON per ablation |

`in1k_finetune_config` and `dinov3_probe_data` read the released checkpoints' `config.json` from
the Hub, or from the directory `$CANVIT_HUB_ROOT` names. `flops` is analytic.

## Layout

| Module | Role |
|---|---|
| `core.py` | `Dataset`, `Figure`, `Diagram` and their runners |
| `run.py` | the CLI |
| `paths.py` | `data/` and `../exports/` |
| `style.py` | policy labels and colors, shared figure layout |
| `stats.py` | bootstrap CIs |
| `ablations/` | the pretraining ablations: table data, reconstruction, downstream evaluations, loss curves |
| `ade20k/` | ADE20K mIoU by policy and canvas grid, mask-size analysis and its figure |
| `in1k.py`, `bench.py`, `comparison.py`, `hf.py` | ImageNet-1k, latency, the main results figure, released configs |
| `flops/` | FLOP counts: CanViT-B and DINOv3 through `canvit_pytorch.flops`, AME and AdaGlimpse here |
| `diagrams/` | diagrams of CanViT's canvas, composed by the Typst sources in `diagrams/*.typ` |

## Citation

```bibtex
@article{berreby2026canvit,
  title={CanViT: Toward Active-Vision Foundation Models},
  author={Berreby, Yoha{\"i}-Eliel and Du, Sabrina and Durand, Audrey and Krishna, B. Suresh},
  year={2026},
  eprint={2603.22570},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.22570}
}
```

## License

MIT. See [LICENSE](LICENSE).
