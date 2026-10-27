# adaglimpse-nnx

AdaGlimpse's architecture (Pardyl et al., ECCV 2024, doi:10.1007/978-3-031-72664-4_7) in JAX/Flax NNX, as the paper's
architecture control (`supp:architecture-control`) pretrains it by distillation from DINOv3. It is not published to
PyPI: `pyproject.toml` carries the `Private :: Do Not Upload` classifier.

## Terms

This directory reimplements the architecture of AdaGlimpse
([apardyl/AdaGlimpse](https://github.com/apardyl/AdaGlimpse)), whose repository carries no license. AdaGlimpse's
model code is modified from MAE ([facebookresearch/mae](https://github.com/facebookresearch/mae)), whose license is
CC BY-NC 4.0, and this port follows that code. Unlike the rest of this repository, this directory is not offered
under the MIT license, and it makes no license grant of its own. Users must respect AdaGlimpse's and MAE's terms.

## What it ports

The model is AdaGlimpse's `MaskedAutoencoderViT`, ported from AdaGlimpse's repository at the commit
`adaglimpse_nnx.upstream` names. Parameter names follow the upstream modules, so each parameter is one tensor of an
AdaGlimpse state dict. `ARCHITECTURE_CONTROL` is the configuration the control trained.

| Module | Holds |
|---|---|
| `config` | `AdaGlimpseConfig` and `ARCHITECTURE_CONTROL` |
| `tokenization` | A glimpse cut into square crops, each resized to a patch, and the scene box of each crop |
| `positional_embedding` | The sine-cosine embeddings of a patch's scene box and of the decoder's output grid |
| `model` | `AdaGlimpseForPretraining`: the encoder, the decoder and the two readouts, and the rollout over glimpses |
| `flops` | Analytic forward FLOPs after each glimpse, and the per-glimpse cost behind the control's ADE20K FLOPs |

AdaGlimpse keeps every patch it has seen and encodes the whole set again after each glimpse; the decoder then
attends over the encoding and a mask token per cell of its output grid. The control reads the decoder's
pixel-reconstruction layer (`decoder_pred`) as a prediction of the teacher's patch features, and its classifier
(`head`) as a prediction of the teacher's CLS token. The ADE20K probe of the control reads the decoded grid
(`AdaGlimpseForPretraining.decode`).

Upstream's training is not ported: its reinforcement-learning policy and harness, its reconstruction losses, and the
parts of its model only they use (patches without scene boxes, padding masks, recorded attention scores). Its
dropout is left out too; the model its harness builds sets every rate to zero. The control's glimpses come from
`canvit_nnx.sample_at_viewpoint`, which samples a crop continuously, so `patch_boxes` cuts the crop box evenly where
upstream's sampler rounds it to whole pixels.

## Use

```python
import flax.nnx as nnx
import jax
from canvit_nnx import Viewpoint, sample_at_viewpoint

from adaglimpse_nnx import ARCHITECTURE_CONTROL, AdaGlimpseForPretraining

model = AdaGlimpseForPretraining(ARCHITECTURE_CONTROL, rngs=nnx.Rngs(0))
size = ARCHITECTURE_CONTROL.scene_size_px
scene = jax.random.normal(jax.random.key(0), (1, size, size, 3))  # an ImageNet-normalized NHWC scene
state = model.init_state(batch_size=1)
viewpoint = Viewpoint.full_scene(batch_size=1)
glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=128)
output = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
features = model.decode(output.latent)  # [1, grid², decoder_embed_dim]
```

## Checks

```bash
cd baselines/adaglimpse-nnx
uv sync
uv run just
```

The tests in `tests/test_upstream_equivalence.py` fetch AdaGlimpse at the pinned commit from GitHub (once, into
`.pytest_cache`) and run its own modules beside this package at `ARCHITECTURE_CONTROL`: the forward pass and both
readouts with the same weights, every parameter's initial distribution, the decoder's grid embedding, the glimpse
tokenization against AdaGlimpse's sampler, and the analytic FLOPs against torch.utils.flop_counter's count of
AdaGlimpse's forward. `tests/test_rollout.py` checks that each patch box is the scene area its patch shows, and
that a compiled rollout matches an eager one.

The distillation trainer that pretrained the control's models, and their checkpoints, are not in this repository.
