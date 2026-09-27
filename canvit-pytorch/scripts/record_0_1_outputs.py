# /// script
# requires-python = ">=3.12"
# dependencies = ["canvit-pytorch==0.1.10", "tyro"]
# ///
"""Record what canvit-pytorch 0.1 computes for ImageNet-1k classifiers in the 0.1 format, on fixed inputs.

    uv run canvit-pytorch/scripts/record_0_1_outputs.py --models DIR [DIR ...] --out reference.pt

A 0.2 conversion of each classifier must reproduce these outputs bit for bit (canvit_pytorch.hub.publish,
"classifier"). The file stores its inputs, and publishing refuses a reference whose inputs differ from
canvit_pytorch.hub.reference's.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from PIL import Image

from canvit_pytorch import CanViTForImageClassification, Viewpoint, sample_at_viewpoint
from canvit_pytorch.preprocess import preprocess

IMAGE = Path(__file__).parents[1] / "test_data" / "Cat03.jpg"
VIEWPOINTS = [(0.0, 0.0, 1.0), (-0.5, 0.5, 0.5), (0.3, -0.2, 0.25)]  # (row, col, scale)
SCENE_SIZE_PX, GLIMPSE_SIZE_PX, CANVAS_GRID_SIZE = 512, 128, 32


@dataclass(frozen=True)
class Args:
    models: list[Path]
    """Classifier directories in the 0.1 format (config.json and model.safetensors)."""
    out: Path


@torch.inference_mode()
def outputs(model_dir: Path) -> dict[str, torch.Tensor | str]:
    model = CanViTForImageClassification.from_pretrained(str(model_dir)).eval()
    image = preprocess(SCENE_SIZE_PX)(Image.open(IMAGE).convert("RGB")).unsqueeze(0)
    state = model.init_state(batch_size=1, canvas_grid_size=CANVAS_GRID_SIZE)
    for row, col, scale in VIEWPOINTS:
        viewpoint = Viewpoint(centers=torch.tensor([[row, col]]), scales=torch.tensor([scale]))
        glimpse = sample_at_viewpoint(spatial=image, viewpoint=viewpoint, glimpse_size_px=GLIMPSE_SIZE_PX)
        logits, state = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
    return {"kind": "classification", "logits": logits, "canvas": state.canvas}


def main(args: Args) -> None:
    inputs = {"image_sha256": hashlib.sha256(IMAGE.read_bytes()).hexdigest(), "viewpoints": VIEWPOINTS,
              "scene_size_px": SCENE_SIZE_PX, "glimpse_size_px": GLIMPSE_SIZE_PX, "canvas_grid_size": CANVAS_GRID_SIZE}
    torch.save({"inputs": inputs, "models": {str(d.resolve()): outputs(d) for d in args.models}}, args.out)
    print(f"wrote {args.out}: {len(args.models)} models")


if __name__ == "__main__":
    main(tyro.cli(Args))
