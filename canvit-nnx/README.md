# canvit-nnx

The JAX and Flax NNX implementation of CanViT (Canvas Vision Transformer). Read the root
[backend documentation](https://github.com/m2b3/CanViT#backend-packages) for
the cross-backend API and tensor conventions.

## Install

```bash
uv add canvit-nnx
```

## Load a Hugging Face checkpoint

Load a pretrained model and an ADE20K probe from Hugging Face. Replace
`path/to/image.jpg` with your image path.

```python
import jax
import jax.numpy as jnp
from PIL import Image

from canvit_nnx import CanViTForSemanticSegmentation, Viewpoint, sample_at_viewpoint
from canvit_nnx.preprocess import preprocess

scene = jnp.asarray(preprocess(512)(Image.open("path/to/image.jpg").convert("RGB")))[None]
model = CanViTForSemanticSegmentation.from_pretrained_with_probe(
    pretrained_repo="canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02-nnx",
    probe_repo="canvit/probe-ade20k-40k-s512-c64-in21k-nnx",
)
model.eval()  # Flax NNX mutates the module and returns None.
viewpoint = Viewpoint.full_scene(batch_size=1)
state = model.init_state(batch_size=1, canvas_grid_size=64)
glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=128)
logits, state = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
jax.block_until_ready(logits)
print(logits.shape)  # [1, 64, 64, 150]
```

The model and probe remain separate checkpoints. They use NHWC arrays: scenes
and glimpses are `[B, H, W, 3]`, and the wrapper returns canvas logits as
`[B, G, G, num_classes]`. `Viewpoint` centers use `(row, col)` scene
coordinates in `[-1, 1]`; each scale is the crop's half-side.

The example uses the [ImageNet-21k CanViT-B checkpoint](https://huggingface.co/canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02-nnx)
and the [ADE20K probe](https://huggingface.co/canvit/probe-ade20k-40k-s512-c64-in21k-nnx).
The [JAX / Flax NNX collection](https://huggingface.co/collections/canvit/canvit-jax-flax-nnx)
also contains the ImageNet-1k pretrained and fine-tuned native checkpoints.

Constructing a model without a checkpoint requires an explicit parameter RNG,
for example:

```python
from flax import nnx
from canvit_nnx import CanViT, CanViTConfig

model = CanViT(CanViTConfig(), rngs=nnx.Rngs(0))
```

## Develop from a checkout

```bash
cd canvit-nnx
uv sync
```

The checkout resolves `canvit-core` from its sibling directory. To convert a
PyTorch checkpoint and run the conversion checks from the repository root:

```bash
uv run --project canvit-nnx python -m tools.convert_checkpoints --help
```
