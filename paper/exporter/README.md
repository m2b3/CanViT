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
uv run python -m canvit_paper_exporter.run         # registered datasets, then figures
uv run python -m canvit_paper_exporter.run NAME... # chosen datasets, figures or diagrams, in order
```

The no-argument run executes the registered datasets, then the figures. Datasets that evaluate ablation
checkpoints and all diagrams are name-only; `run list` prints their names. Run a dataset before any figure that
depends on its JSON. Diagrams load the released CanViT-B from the Hugging Face Hub and require `typst` on `PATH`.

Figures render with matplotlib's Agg backend and the TeX Gyre Heros font that ultraplot bundles;
`run_figure` fails if a PDF embeds any other font.

## Inputs

Evaluation results live under `data/` (not tracked). The commands in the last column run from `canvit-pytorch/`
and place the files where this exporter reads them:

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
| `ablation_variants`, `ablation_loss` | `comet/` | pretraining curves exported from Comet, one JSON per ablation |

`in1k_finetune_config` and `dinov3_probe_data` read the released checkpoints' `config.json` from
the Hub, or from the directory `$CANVIT_HUB_ROOT` names. `flops` is analytic.

Figures use these generated datasets:

| Figure | Required datasets |
|---|---|
| `ablation_loss` | `ablation_variants`, plus the per-ablation JSON files in `comet/` |
| `resolution_and_mask_size` | `ade20k_iou_vs_obj`, `ade20k_seg` |
| `canvas_grid_impact_coarse_to_fine` | `ade20k_seg` |
| `hw_latency` | `hw_bench` |
| `main_results` | `ade20k_seg`, `in1k_clf_frozen`, `in1k_clf_finetuned` |
| `canvas_attention_rhs` | none; analytic FLOP formulas |

## Citation

See the [project citation](../../README.md#citation).

## License

MIT. See [LICENSE](../../LICENSE).
