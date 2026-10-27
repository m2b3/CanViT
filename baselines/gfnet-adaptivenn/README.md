# gfnet-adaptivenn

The measurements of GFNet (Wang et al., NeurIPS 2020) and AdaptiveNN (Wang et al., Nature Machine Intelligence 2025)
behind the paper's ImageNet-1k comparison at a fixed number of glimpses (`supp:in1k-efficiency`), made with each
model's released code and checkpoints. It is not published to PyPI: `pyproject.toml` carries the
`Private :: Do Not Upload` classifier.

Every image takes every glimpse: GFNet runs its "anytime prediction" evaluation and AdaptiveNN takes all its
fixations, so neither exits early. FLOPs count matrix products and convolutions with torch.utils.flop_counter over
each model's own inference code, at 2 FLOPs per multiply-accumulate, policy networks and classifiers included.

## Sources

`gfnet_adaptivenn.sources` pins each repository by commit and each checkpoint by SHA-256, with the Google Drive link
of the repository's README. GFNet's code imports `torch._six`, which PyTorch 2.0 removed; the checkout replaces that
import with `collections.abc`, which `torch._six` re-exported, and every command checks that the checkout differs
from the commit by that line alone.

```bash
cd baselines/gfnet-adaptivenn
uv sync
uv run python -m gfnet_adaptivenn.sources --root sources
```

## Measure

The paper's exporter reads each record under the name `canvit_paper_exporter.in1k_efficiency` gives it
(`GFNET_FACTS`, `GFNET_ACCURACY`, `GFNET_FLOPS`, `ADAPTIVENN_ACCURACY`, `ADAPTIVENN_FLOPS`, `ACTIVE_PARAMS`).

| Command | Record | Needs |
|---|---|---|
| `gfnet_adaptivenn.gfnet facts` | What each GFNet checkpoint stores: configuration, its authors' cumulative MACs and top-1, the size of each state dict | CPU |
| `gfnet_adaptivenn.gfnet accuracy` | GFNet's top-1 after each glimpse, from its `inference.py --eval_mode 1` | CUDA, ImageNet-1k val |
| `gfnet_adaptivenn.gfnet flops` | GFNet's cumulative GFLOPs over its inference loop | CUDA |
| `gfnet_adaptivenn.adaptivenn accuracy` | AdaptiveNN's top-1 after the glance and each fixation, from its evaluation command | CUDA, ImageNet-1k val |
| `gfnet_adaptivenn.adaptivenn flops` | AdaptiveNN's cumulative GFLOPs by number of fixations | CPU |
| `gfnet_adaptivenn.active_params` | `parameters()` of the modules each model runs up to each glimpse, classifiers included, named from its inference code; buffers and training-only modules are not counted | CPU |

```bash
uv run python -m gfnet_adaptivenn.gfnet accuracy --root sources --imagenet-val /path/to/imagenet/val \
  --out records/gfnet_accuracy.json
```

ImageNet-1k's validation images are in ImageFolder layout, a directory per class. `--help` on each command lists
its options. The accuracy commands write each model's own output beside the record, as `<checkpoint>.log`.

## Checks

```bash
uv run just                 # lint, types
uv run just test sources    # the parameter record against counts derived separately; needs the fetched root
```
