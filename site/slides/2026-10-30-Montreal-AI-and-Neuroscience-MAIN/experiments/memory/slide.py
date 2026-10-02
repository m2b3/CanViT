"""The memory slide's images for an exported sequence: <export>/slide/{kept,reset}_t<t>.png (draw.composite at 512 px)
and legend.json (each class with its color, in glimpse order). The scene with the boxes so far is the export's
scene_t<t>.png."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import tyro
from PIL import ImageFont

from experiments import logs
from experiments.memory.draw import CLASS_COLORS, panels

log = logging.getLogger(__name__)

PANEL = 512  # px, a multiple of the canvas grid


@dataclass(frozen=True)
class Config:
    export: Path
    """a directory export.py wrote"""


def main(cfg: Config) -> None:
    record, drawn = panels(cfg.export, ImageFont.load_default(size=28), PANEL)
    out = cfg.export / "slide"
    out.mkdir(exist_ok=True)
    for name, image in drawn.items():
        if name != "scene":
            image.save(out / f"{name}.png")
    legend = [{"class": o["class"], "color": c} for o, c in zip(record["objects"], CLASS_COLORS, strict=True)]
    (out / "legend.json").write_text(json.dumps(legend, indent=1))
    log.info("%s: %s", out, ", ".join(o["class"] for o in record["objects"]))


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
