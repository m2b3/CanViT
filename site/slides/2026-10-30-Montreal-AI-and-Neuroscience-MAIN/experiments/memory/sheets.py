"""Contact sheets of ranked, exported sequences, for choosing the slide's example; no model is run here. Reads the
ranking for the order and the numbers, and the exports for the pixels. Writes into <out>/:
  sheet-<n>.png       SHEET_ROWS x SHEET_COLUMNS candidates each: the scene with the three glimpse boxes (numbered in
                      their object's color), the composite after the third glimpse with the canvas kept, and with the
                      canvas reset before each glimpse
  build-<slug>.png    per candidate, the kept composite after each glimpse, then the reset one after each glimpse"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import tyro
from PIL import Image, ImageDraw, ImageFont

from experiments import logs
from experiments.memory.draw import CLASS_COLORS, panels
from experiments.outputs import WORK

log = logging.getLogger(__name__)

PANEL = 256  # px, a multiple of the canvas grid
GAP, LABEL_H = 6, 34
SHEET_COLUMNS, SHEET_ROWS = 2, 4


@dataclass(frozen=True)
class Config:
    exports_root: Path
    """the directory export.py wrote the ranked sequences into"""
    ranking: Path = WORK / "memory/ranking.json"
    out: Path = WORK / "memory/figures"
    top: int = 24


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], rank: int, entry: dict, font: ImageFont.ImageFont) -> None:
    x, y = xy
    draw.text((x, y), f"#{rank} {entry['image_id']}  {entry['category']}  {entry['short_side_px']}px  score "
                      f"{entry['score']:.2f}", fill="black", font=font)
    x_text = x
    for i, o in enumerate(entry["objects"]):
        text = f"{o['name']} {o['seen']:.2f}/{o['kept']:.2f}/{o['reset']:.2f}  "
        draw.rectangle([x_text, y + 18, x_text + 10, y + 28], fill=CLASS_COLORS[i])
        draw.text((x_text + 14, y + 16), text, fill="black", font=font)
        x_text += 14 + int(draw.textlength(text, font=font))


def main(cfg: Config) -> None:
    ranking = json.loads(cfg.ranking.read_text())["top"][:cfg.top]
    cfg.out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=13)
    per_sheet = SHEET_COLUMNS * SHEET_ROWS
    cell_w, cell_h = 3 * PANEL + 2 * GAP, LABEL_H + PANEL
    for sheet_index, start in enumerate(range(0, len(ranking), per_sheet)):
        chunk = ranking[start:start + per_sheet]
        sheet = Image.new("RGB", (SHEET_COLUMNS * (cell_w + 3 * GAP), SHEET_ROWS * (cell_h + 2 * GAP)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, entry in enumerate(chunk):
            record, drawn = panels(cfg.exports_root / entry["slug"], font, PANEL)
            x, y = (k % SHEET_COLUMNS) * (cell_w + 3 * GAP), (k // SHEET_COLUMNS) * (cell_h + 2 * GAP)
            label(draw, (x, y), start + k + 1, entry, font)
            last = len(record["viewpoints"]) - 1
            for j, name in enumerate(("scene", f"kept_t{last}", f"reset_t{last}")):
                sheet.paste(drawn[name], (x + j * (PANEL + GAP), y + LABEL_H))
            build = Image.new("RGB", (len(drawn) * (PANEL + GAP), PANEL + LABEL_H), "white")
            label(ImageDraw.Draw(build), (0, 0), start + k + 1, entry, font)
            for j, image in enumerate(drawn.values()):
                build.paste(image, (j * (PANEL + GAP), LABEL_H))
            build.save(cfg.out / f"build-{entry['slug']}.png")
        target = cfg.out / f"sheet-{sheet_index}.png"
        sheet.save(target)
        log.info("%s", target)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
