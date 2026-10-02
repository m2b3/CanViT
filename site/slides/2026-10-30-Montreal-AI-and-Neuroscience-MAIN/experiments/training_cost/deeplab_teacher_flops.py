"""Forward FLOPs of AdaGlimpse's segmentation teacher, DeepLabV3-ResNet101 at 224 px with 150 classes, counted by
torch's FlopCounterMode; baselines.py's TEACHER_DEEPLAB_FWD is this number. From the talk's directory:

    uv run --no-sync --project ../../../canvit-pytorch python -m experiments.training_cost.deeplab_teacher_flops
"""

import inspect

import torch
import torchvision
from torch.utils.flop_counter import FlopCounterMode
from torchvision.models.segmentation import deeplabv3_resnet101

print("torch", torch.__version__, "torchvision", torchvision.__version__)
print(inspect.signature(deeplabv3_resnet101))

# AdaGlimpse rl_glimpse.py:137 @ e5f37fd: deeplabv3_resnet101(num_classes=self.decoder_out_channels).
# weights_backbone only changes initial values, not the graph; None avoids a download.
model = deeplabv3_resnet101(num_classes=150, weights_backbone=None).eval()
assert model.aux_classifier is None, "aux head would add FLOPs"
x = torch.zeros(1, 3, 224, 224)
with torch.no_grad(), FlopCounterMode(display=False) as counter:
    out = model(x)["out"]
assert out.shape == (1, 150, 224, 224), out.shape
total = counter.get_total_flops()
by_op = {str(k): v for k, v in counter.get_flop_counts()["Global"].items()}
print(f"DeepLabV3-R101 @224, 150 classes, forward: {total / 1e9:.3f} GFLOPs (2 FLOPs per MAC)")
print("by op:", {k: f"{v / 1e9:.3f} GF" for k, v in by_op.items()})
