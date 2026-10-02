# FLOPs (`canvit_pytorch.flops`)

`canvit_pytorch.flops` computes analytic forward floating-point operation
counts for CanViT and the DINOv3 ViTs used in the evaluations. CanViT counts
come from a `CanViTConfig` and its backbone specification. DINOv3 counts come
from a `DINOv3ViTConfig` loaded from the model configuration.

## Setup

The CanViT functions use the base package. The DINOv3 example and the network
test need the optional transformer dependency:

```bash
uv sync --all-extras
```

## Counting convention

One multiply-accumulate counts as two FLOPs. The functions count the matrix
products and convolutions covered by `torch.utils.flop_counter`:

- every linear layer and patch-embedding convolution;
- both products in each attention operation, with
  `4 * num_queries * num_keys * dimension` FLOPs over all heads;
- the Viewpoint Encoding (VPE) random Fourier feature projection; and
- the segmentation probe's 1×1 convolution.

Normalization, softmax, rotary position embeddings (RoPE), activations,
LayerScale, residual additions, glimpse extraction and viewing-policy work are
outside the count. CanViT counts are per glimpse. A rollout with `T` glimpses
of the same geometry therefore performs `T` times this CanViT count before any
downstream readout.

| Function | Count |
| --- | --- |
| `glimpse_flops` | One CanViT forward, including patch embedding, VPE, backbone blocks, and scheduled Canvas Attention Reads and Writes. |
| `canvas_attention_read_flops` | One Canvas Attention Read. |
| `canvas_attention_write_flops` | One Canvas Attention Write. |
| `segmentation_probe_flops` | A `SegmentationProbe` on one feature grid. |
| `dinov3_flops` | One DINOv3 ViT forward on a square RGB image. |
| `num_glimpse_tokens` | The number of tokens in the glimpse stream. |
| `num_canvas_tokens` | The number of canvas registers and canvas patches. |

For the paper's ADE20K per-glimpse cost, add the CanViT count to the linear
segmentation probe count:

```python
from canvit_pytorch import CanViTConfig
from canvit_pytorch.benchmarks.ade20k import NUM_CLASSES
from canvit_pytorch.flops import glimpse_flops, segmentation_probe_flops

config = CanViTConfig()
canvit = glimpse_flops(config, glimpse_size_px=128, canvas_grid_size=32)
probe = segmentation_probe_flops(
    grid_size=32,
    embed_dim=config.canvas_dim,
    num_classes=NUM_CLASSES,
)
print(f"{(canvit + probe) / 1e9:.1f} GFLOPs per glimpse")
```

Load a DINOv3 configuration before calling `dinov3_flops`:

```python
from transformers import AutoConfig

from canvit_pytorch.flops import dinov3_flops
from canvit_pytorch.teacher import DINOV3_REPOS

config = AutoConfig.from_pretrained(DINOV3_REPOS["vitb16"])
print(dinov3_flops(config, input_size_px=512))
```

The model configuration repositories are gated on the [Hugging Face
Hub](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) and
require an accepted license and a token.

## Tests

```bash
uv run --all-extras pytest -q tests/test_flops.py
```

The tests compare the analytic counts with real forwards measured by
`torch.utils.flop_counter` under the math attention backend. They also cover
CanViT's released geometry, its ablations and DINOv3 configurations.
