"""The glimpses CanViT receives on a tour of a still scene (#next): at each viewpoint, the 128 px glimpse
(sample_at_viewpoint), with the viewpoint's box as fractions of the scene. The tour moves and zooms in and out.

    uv run --no-sync --project ../../../canvit-pytorch python -u -m experiments.glimpse_tour.export

Writes ../../data/talk/glimpse-tour/: NN.png (each glimpse) and manifest.json (the scene, the glimpse size, the patch
grid, and each stop's glimpse file, viewpoint and box).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.hub.repos import RELEASED_GLIMPSE_SIZE_PX, RELEASED_SCENE_SIZE_PX
from canvit_pytorch.preprocess import imagenet_denormalize_uint8, imagenet_normalize
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint
from PIL import Image

HERE = Path(__file__).resolve().parent
SITE = HERE.parents[3]

# (row, col, scale), centers in [-1, 1] as tensor indexing: the whole street, then the scooter rider and the car, the
# cafe's sign, the hotel's sign up the facade, the far side of the street, and the street's middle at mid zoom. The
# points are #human-vision's fixations on the same scene.
TOUR = (
    (0.0, 0.0, 1.0),
    (0.3, -0.36, 0.6),
    (-0.18, -0.76, 0.2),
    (-0.62, -0.22, 0.25),
    (0.24, 0.6, 0.35),
    (0.0, -0.1, 0.5),
)


@dataclass(frozen=True)
class Export:
    scene: Path = SITE / "data/street-riid/scene.png"
    out: Path = SITE / "data/talk/glimpse-tour"


def main(args: Export) -> None:
    pixels = np.array(Image.open(args.scene).convert("RGB"))
    if pixels.shape != (RELEASED_SCENE_SIZE_PX, RELEASED_SCENE_SIZE_PX, 3):
        raise ValueError(f"glimpse_tour: {args.scene} is {pixels.shape}, not a {RELEASED_SCENE_SIZE_PX} px RGB scene")
    scene = imagenet_normalize(torch.from_numpy(pixels).permute(2, 0, 1).float() / 255)[None]
    args.out.mkdir(parents=True, exist_ok=True)
    stops = []
    for i, (row, col, scale) in enumerate(TOUR):
        if not (0 < scale <= 1 and abs(row) <= 1 - scale and abs(col) <= 1 - scale):
            raise ValueError(f"glimpse_tour: viewpoint {(row, col, scale)} leaves the scene")
        viewpoint = Viewpoint(centers=torch.tensor([[row, col]]), scales=torch.tensor([scale]))
        glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)[0]
        name = f"{i:02d}.png"
        Image.fromarray(imagenet_denormalize_uint8(glimpse)).save(args.out / name)
        stops.append({"glimpse": name, "viewpoint": {"row": row, "col": col, "scale": scale},
                      "box": {"top": (row - scale + 1) / 2, "left": (col - scale + 1) / 2, "size": scale}})
    (args.out / "manifest.json").write_text(json.dumps({
        "scene": "ADE20K validation (data/street-riid/scene.png)", "glimpse_px": RELEASED_GLIMPSE_SIZE_PX,
        "patch_grid": 8, "stops": stops}, indent=1))
    print(args.out, len(stops))


if __name__ == "__main__":
    main(tyro.cli(Export))
