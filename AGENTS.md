# CanViT: guide for contributors and coding agents

`CLAUDE.md` is a symlink to this file.

## Layout: one home per thing

- `canvit-pytorch/`: the Python distribution `canvit-pytorch` (import
  `canvit_pytorch`). The model lives at the package top level; subsystems are
  verbs: `canvit_pytorch.pretrain`, `.specialize` (probes, fine-tuning),
  `.evaluate` (benchmarks). Tests in `canvit-pytorch/tests/` and in
  `test.py` files beside the code; docs in `canvit-pytorch/docs/`; SLURM job
  scripts in `canvit-pytorch/slurm/`; one-off scripts in
  `canvit-pytorch/scripts/`.
- `canvit-pytorch/tpu/`: a separate uv environment for ImageNet-1k
  fine-tuning on Cloud TPU (exact torch/torch_xla pins).
- `site/`: the project page, deployed to https://m2b3.github.io/CanViT/ by
  `.github/workflows/pages.yml` on pushes to `main` that touch it.
- `.github/workflows/release.yml`: PyPI release of `canvit-pytorch` on `v*`
  tags.

Before adding a file, find the existing home for its kind of content; a fact,
constant or URL lives in one place and everything else points to it (for
example, repository and install URLs come from `canvit_pytorch.checkpoints`).

## Commands

From `canvit-pytorch/`:

```bash
uv sync --all-extras   # a plain `uv sync` is exact and removes the extras
uv run pypatree        # module tree, a good first look
uv run just            # lint, typecheck, test
```

## Conventions that are easy to get wrong

- Viewpoint centers are `(row, col)` in `[-1, 1]`, matching tensor indexing,
  not Cartesian `(x, y)`; scale `s` is the crop's half side, so a glimpse
  covers `s²` of the scene.
- Standard canvas grid: 32×32 tokens; patch size 16 px; glimpses 128 px.
- `torch.compile`: call `model(x)`, never `model.forward(x)`, which bypasses
  the compiled wrapper.
- Numbers reported anywhere (README, site, papers) come from saved evaluation
  outputs, never typed by hand.

## Git

Never `git add -A` or `git add .`; stage files by name and read the staged
diff before committing.
