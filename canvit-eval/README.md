# CanViT-eval

Evaluation and benchmarking for [CanViT](https://github.com/m2b3/CanViT-PyTorch),
the Canvas Vision Transformer.

## Install

Requires [`uv`](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```

## Datasets

ADE20K (`ADEChallengeData2016/`) and ImageNet-1k (`ILSVRC2012/val/`) are
referenced depending on the eval. Three ways to point `canvit-eval` at them:

1. Export in your shell:

   ```bash
   export ADE20K_ROOT=/path/to/ADEChallengeData2016
   export IMAGENET_VAL=/path/to/ILSVRC2012/val
   ```

2. Copy `.envrc.example` to `.envrc`, edit, then `source .envrc` per shell or
   use [direnv](https://direnv.net/) to auto-load on `cd`.

3. Pass paths per task on the CLI (run `--help` on the subcommand for the
   exact flag).

## Single eval

```bash
uv run python -m canvit_eval --help                       # list subcommands
uv run python -m canvit_eval ade20k-seg-canvit --help     # full flag set
```

Subcommands:

- `ade20k-seg-canvit`: ADE20K mIoU via a CanViT episode (T-step rollout, mIoU
  per timestep).
- `ade20k-seg-dinov3`: ADE20K mIoU with a passive DINOv3 backbone (single
  forward at a fixed input resolution, mIoU at t=0). Baseline.
- `in1k-clf`: ImageNet-1k top-k classification, fused-frozen-probe or
  finetuned.
- `reconstruction`: cosine similarity between CanViT canvas/CLS and DINOv3
  teacher features per timestep.

Concrete example (flagship ADE20K config: 512 px scene, 64×64 canvas,
21 timesteps):

```bash
uv run python -m canvit_eval ade20k-seg-canvit \
    --probe-repo canvit/probe-ade20k-40k-s512-c64-in21k \
    --scene-size 512 --episode.canvas-grid 64 \
    --output results/ade20k_seg.pt
```

Saves a `.pt` with per-timestep mIoU and run metadata.

## Batch eval

```bash
uv run python -m canvit_eval.batch --help
uv run python -m canvit_eval.batch --n-runs 5
```

Sweeps the four subcommands above across a predefined set of
(scene size, canvas grid, policy) configurations, one after another.
`--n-runs` sets the repetition count for stochastic policies; deterministic
ones always run once. `--include-extra-grids` adds canvas-grid sweeps beyond
the baseline set. `--skip-existing` skips configs whose output files already
exist.

## ADE20K mask-size pipeline

Three stages: DINOv3 feature export, DINOv3 IoU, CanViT IoU. Each stage skips
if its output already exists.

```bash
uv run python -m canvit_eval.tasks.ade20k_obj                              # all stages
uv run python -m canvit_eval.tasks.ade20k_obj.export_dv3_features --help   # stage 1 alone
uv run python -m canvit_eval.tasks.ade20k_obj.iou --help                   # stages 2 & 3
```

## Latency bench

```bash
uv run python bench/pt/matrix.py --help                                    # bench across many configs
uv run python bench/pt/run.py --help                                       # bench one config
uv run python bench/pt/analyze.py --pattern 'bench/pt/results/*.jsonl'     # summarise the JSONLs
```

## Tests

```bash
uv run pytest
```

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

MIT. See [LICENSE](LICENSE) for details.
