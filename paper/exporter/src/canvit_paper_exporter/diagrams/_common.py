"""Configuration and paths shared by the diagram generators.

Each generator writes PNGs under diagrams/outputs/<name>/; diagrams/<name>.typ composes them into
exports/<name>.svg and .pdf.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from canvit_pytorch import CanViT, CanViTForPretraining
from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_CANVAS_GRID_SIZE, RELEASED_GLIMPSE_SIZE_PX

from canvit_paper_exporter.paths import PAPER_ROOT, REPO_ROOT

DIAGRAMS_DIR = REPO_ROOT / "diagrams"
DIAGRAMS_OUTPUTS = DIAGRAMS_DIR / "outputs"
# The repository's example images, which canvit-pytorch's demos and tests also read.
EXAMPLE_IMAGES = PAPER_ROOT.parent / "canvit-pytorch" / "test_data"


@dataclass
class BaseConfig:
    model: str = FLAGSHIP
    image: Path = EXAMPLE_IMAGES / "Cat03.jpg"
    canvas_grid: int = RELEASED_CANVAS_GRID_SIZE
    glimpse_px: int = RELEASED_GLIMPSE_SIZE_PX
    device: str = "cpu"


def load_canvit(repo: str, device: torch.device) -> CanViT:
    """The pretrained CanViT of a checkpoint, without its pretraining readouts."""
    return CanViTForPretraining.from_pretrained(repo).canvit.to(device).eval()
