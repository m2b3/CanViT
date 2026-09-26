"""Classify an image glimpse by glimpse with CanViT-B, printing the top-1 class after each glimpse.

Two released classifiers share one interface: the model fine-tuned on
ImageNet-1k, and the frozen pretrained model whose CLS readout is fused with a
linear probe fitted on its DINOv3 teacher.

    uv run --extra demo python demos/classify.py
    uv run --extra demo python demos/classify.py --image path/to/image.jpg --classifier frozen
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from PIL import Image

from canvit_pytorch import CanViTForImageClassification
from canvit_pytorch.benchmarks.imagenet import CLASS_NAMES
from canvit_pytorch.hub.repos import (
    DINOV3_VITB16_IN1K_PROBE,
    FINETUNED_IN1K,
    FLAGSHIP,
    RELEASED_CANVAS_GRID_SIZE,
    RELEASED_GLIMPSE_SIZE_PX,
    RELEASED_SCENE_SIZE_PX,
)
from canvit_pytorch.policies import POLICIES, PolicyName, make_policy
from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.viewpoint import sample_at_viewpoint


@dataclass(frozen=True)
class Demo:
    image: Path = Path("test_data/Cat03.jpg")
    classifier: Literal["finetuned", "frozen"] = "finetuned"
    policy: PolicyName = "coarse_to_fine"
    num_glimpses: int = 5
    scene_size_px: int = RELEASED_SCENE_SIZE_PX
    canvas_grid_size: int = RELEASED_CANVAS_GRID_SIZE
    glimpse_size_px: int = RELEASED_GLIMPSE_SIZE_PX
    seed: int = 0


def load(classifier: Literal["finetuned", "frozen"]) -> CanViTForImageClassification:
    if classifier == "finetuned":
        return CanViTForImageClassification.from_pretrained(FINETUNED_IN1K)
    return CanViTForImageClassification.from_pretrained_with_probe(pretrained_repo=FLAGSHIP, probe_repo=DINOV3_VITB16_IN1K_PROBE)


@torch.inference_mode()
def main(demo: Demo) -> None:
    torch.manual_seed(demo.seed)
    model = load(demo.classifier).eval()
    image = preprocess(demo.scene_size_px)(Image.open(demo.image).convert("RGB"))
    assert isinstance(image, torch.Tensor)
    image = image.unsqueeze(0)
    policy = make_policy(demo.policy, batch_size=1, device=image.device, num_glimpses=demo.num_glimpses,
                         canvas_grid_size=demo.canvas_grid_size)
    state = model.init_state(batch_size=1, canvas_grid_size=demo.canvas_grid_size)
    print(f"{demo.image}, {demo.classifier} classifier, {POLICIES[demo.policy].paper_name}")
    for t in range(demo.num_glimpses):
        viewpoint = policy.step(t, state)
        glimpse = sample_at_viewpoint(spatial=image, viewpoint=viewpoint, glimpse_size_px=demo.glimpse_size_px)
        logits, state = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
        probability, index = torch.softmax(logits[0], dim=-1).max(dim=-1)
        (row, col), scale = viewpoint.centers[0].tolist(), viewpoint.scales[0].item()
        print(f"t={t}  row {row:+.2f} col {col:+.2f} scale {scale:.2f}  ->  {CLASS_NAMES[int(index)]} ({probability:.1%})")


if __name__ == "__main__":
    main(tyro.cli(Demo))
