"""Composites of an exported sequence (export.py): each object's class probability in its own color over the scene in
dimmed grayscale, opacity ALPHA * p; the maps are per canvas cell, upsampled nearest only."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.memory.definition import covering_box_px

CLASS_COLORS = ("#fde047", "#e879f9", "#22c55e")  # yellow, fuchsia, green, in glimpse order: away from the talk's
# glimpse blue #2d6cdf, canvas red #e0483e and policy teal #0d9488 (dataviz validator, dark gray surface: worst
# all-pairs CVD dE 13.4, normal-vision dE 24.1)
GLIMPSE_BLUE = "#2d6cdf"
DIM = 0.4  # gray level of the scene under the maps
ALPHA = 0.9  # opacity of a class color where p = 1


def rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255


def probability(path: Path, panel: int) -> np.ndarray:
    p = np.asarray(Image.open(path), dtype=np.float32) / 65535
    assert p.ndim == 2 and p.shape[0] == p.shape[1] and panel % p.shape[0] == 0, (path, p.shape)
    repeat = panel // p.shape[0]
    return p.repeat(repeat, axis=0).repeat(repeat, axis=1)


def photo(path: Path, panel: int) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((panel, panel), Image.Resampling.LANCZOS),
                      dtype=np.float32) / 255


def composite(scene: np.ndarray, maps: list[np.ndarray]) -> Image.Image:
    gray = (scene @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
    out = np.repeat(gray * DIM, 3, axis=-1)
    for p, color in zip(maps, CLASS_COLORS):
        alpha = ALPHA * p[..., None]
        out = out * (1 - alpha) + rgb(color) * alpha
    return Image.fromarray((out * 255).round().astype(np.uint8))


def boxed(scene: np.ndarray, record: dict, font: ImageFont.ImageFont, panel: int) -> Image.Image:
    image = Image.fromarray((scene * 255).round().astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for i, vp in enumerate(record["viewpoints"]):
        top, left, bottom, right = covering_box_px((vp["row"], vp["col"], vp["scale"]), panel)
        draw.rectangle([left, top, right - 1, bottom - 1], outline=GLIMPSE_BLUE, width=3)
        draw.rectangle([left, top, left + 16, top + 18], fill=CLASS_COLORS[i])
        draw.text((left + 4, top + 1), str(i + 1), fill="black", font=font)
    return image


def panels(export: Path, font: ImageFont.ImageFont, panel: int) -> tuple[dict, dict[str, Image.Image]]:
    """The export's record, and its panels: the scene with numbered boxes, then the composite after each glimpse, the
    canvas kept (kept_t<t>) and reset (reset_t<t>)."""
    record = json.loads((export / "memory.json").read_text())
    scene = photo(export / "scene.png", panel)
    maps = lambda cond, t: [probability(export / f"{o['file_prefix']}_{cond}_t{t}.png", panel)  # noqa: E731
                            for o in record["objects"]]
    drawn = {"scene": boxed(scene, record, font, panel)}
    for cond in ("kept", "reset"):
        for t in range(len(record["viewpoints"])):
            drawn[f"{cond}_t{t}"] = composite(scene, maps(cond, t))
    return record, drawn
