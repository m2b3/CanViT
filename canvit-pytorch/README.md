# canvit-pytorch

The PyTorch implementation of CanViT (Canvas Vision Transformer). It
contains the model, checkpoint loading, preprocessing, policies, pretraining,
task specialization, and evaluation code. The paper is
[CanViT: Toward Active-Vision Foundation Models](https://arxiv.org/abs/2603.22570) (NeurIPS 2026).

## Install

```bash
uv add canvit-pytorch
```

## Load a Hugging Face checkpoint

This example loads a PyTorch checkpoint, processes a user-provided image, and
returns the recurrent state after one glimpse. Replace the image path with an
RGB image available to your environment.

```python
import torch
from PIL import Image

from canvit_pytorch import CanViTForPretraining, Viewpoint, sample_at_viewpoint
from canvit_pytorch.preprocess import preprocess

scene = preprocess(512)(Image.open("path/to/image.jpg").convert("RGB")).unsqueeze(0)
model = CanViTForPretraining.from_pretrained(
    "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
).eval()
viewpoint = Viewpoint.full_scene(batch_size=1, device=scene.device)
state = model.init_state(batch_size=1, canvas_grid_size=32)
glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=128)

with torch.inference_mode():
    output = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
print(output.state.recurrent_cls.shape)  # [1, 1, 768]
```

The example uses the [ImageNet-21k pretrained CanViT-B checkpoint](https://huggingface.co/canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02).
The [fine-tuned ImageNet-1k classifier](https://huggingface.co/canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06)
and separate task probes are also available from the [PyTorch checkpoints](https://huggingface.co/canvit).
The root [README](https://github.com/m2b3/CanViT) documents classification,
segmentation, native backends, and the training and evaluation entry points.
PyTorch inputs use NCHW: scenes and feature maps are
`[B, C, H, W]`, while model glimpses are `[B, 3, h, w]`. Viewpoint centers use
`(row, col)` scene coordinates in `[-1, 1]`; each scale is the crop's
half-side.

MIT license. See [LICENSE](https://github.com/m2b3/CanViT/blob/main/LICENSE).
