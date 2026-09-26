# FLOPs (`canvit_pytorch.flops`)

Analytic forward FLOP counts of [CanViT](../../README.md), the Canvas Vision
Transformer, and of the DINOv3 ViTs it is compared with. CanViT's come from a
`CanViTConfig` and the backbone it names, DINOv3's from the model's Hugging
Face config.

## Conventions

- One multiply-accumulate is two FLOPs.
- Forward passes only. CanViT's costs are per glimpse; a rollout of T glimpses
  costs T times as much.
- Counted: what `torch.utils.flop_counter` counts, matrix products and
  convolutions. That is every linear layer, both products of each attention
  (4 N_q N_k D for N_q queries, N_k keys and width D over all heads), the patch
  embedding, the Viewpoint Encoding's random Fourier features and a
  segmentation probe's 1×1 convolution.
- Left out: normalization (LayerNorm, BatchNorm), softmax, RoPE, GELU,
  LayerScale, residual additions, glimpse extraction and the viewing policy.

| Function | Cost of |
|---|---|
| `glimpse_flops` | One CanViT forward: patch embedding, VPE, backbone blocks, Canvas Attention |
| `canvas_attention_read_flops`, `canvas_attention_write_flops` | One Canvas Attention Read, one Write |
| `segmentation_probe_flops` | A `SegmentationProbe` on a feature grid |
| `dinov3_flops` | A DINOv3 ViT on a square image |
| `num_glimpse_tokens`, `num_canvas_tokens` | Token counts N_g and N_can (no FLOPs) |

The paper's per-glimpse costs of CanViT, in its ADE20K results and its
ablation table, are `glimpse_flops` plus `segmentation_probe_flops` for the
ADE20K probe on the canvas.

```python
from canvit_pytorch import CanViTConfig
from canvit_pytorch.benchmarks.ade20k import NUM_CLASSES
from canvit_pytorch.flops import glimpse_flops, segmentation_probe_flops

config = CanViTConfig()  # CanViT-B
glimpse = glimpse_flops(config, glimpse_size_px=128, canvas_grid_size=32)
probe = segmentation_probe_flops(grid_size=32, embed_dim=config.canvas_dim, num_classes=NUM_CLASSES)
print(f"{(glimpse + probe) / 1e9:.1f} GFLOPs per glimpse")
```

DINOv3's config comes from the Hub, where the repositories are gated: accept
their license there first.

```python
from transformers import AutoConfig
from canvit_pytorch.flops import dinov3_flops
from canvit_pytorch.teacher import DINOV3_REPOS

dinov3_flops(AutoConfig.from_pretrained(DINOV3_REPOS["vitb16"]), input_size_px=512)
```

## Tests

```bash
uv run --all-extras pytest -q tests/test_flops.py
```

They check every count against `torch.utils.flop_counter` on real forwards
(with the math attention backend, as the counter misses fused CPU attention),
and reproduce the paper's FLOP counts of CanViT-B, its ablations and DINOv3.
