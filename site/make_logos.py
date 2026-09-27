# /// script
# requires-python = ">=3.12"
# dependencies = ["fonttools", "uharfbuzz", "tyro"]
# ///
"""Writes the logos that are generated rather than downloaded (see README.md, "Logo sources").

    uv run site/make_logos.py --font /path/to/Inter-4.1/extras/ttf/InterDisplay-ExtraBold.ttf

- assets/logos/canvit-wordmark.svg: "CanViT" as outlines, the title's font and letter-spacing, filled with the
  page's gradient from the canvas's red to the glimpse's blue.
- assets/logos/neurips-dark.svg: the NeurIPS logo with its dark gray text made light, for dark backgrounds.
"""

from dataclasses import dataclass
from pathlib import Path

import tyro
import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

LOGOS = Path(__file__).parent / "assets/logos"
WORDMARK = "CanViT"
# style.css: the title's letter-spacing, --canvas and --glimpse.
LETTER_SPACING_EM = -0.022
CANVAS_RED = "#e0483e"
GLIMPSE_BLUE = "#2d6cdf"
PADDING_EM = 0.02
NEURIPS_TEXT_FILL = 'fill="#323539"'
NEURIPS_TEXT_FILL_ON_DARK = 'fill="#e6edf3"'


@dataclass
class Args:
    font: Path
    """InterDisplay-ExtraBold.ttf from Inter 4.1 (github.com/rsms/inter/releases/tag/v4.1)."""


def wordmark_svg(font_path: Path) -> str:
    face = hb.Face(hb.Blob.from_file_path(str(font_path)))
    buffer = hb.Buffer()
    buffer.add_str(WORDMARK)
    buffer.guess_segment_properties()
    hb.shape(hb.Font(face), buffer, {"kern": True})

    font = TTFont(font_path)
    glyphs, order, upem = font.getGlyphSet(), font.getGlyphOrder(), face.upem
    paths, bounds, x = [], BoundsPen(glyphs), 0.0
    for info, position in zip(buffer.glyph_infos, buffer.glyph_positions, strict=True):
        name = order[info.codepoint]
        # Font units point up; SVG's point down.
        transform = (1, 0, 0, -1, x + position.x_offset, -position.y_offset)
        path = SVGPathPen(glyphs)
        glyphs[name].draw(TransformPen(path, transform))
        glyphs[name].draw(TransformPen(bounds, transform))
        paths.append(path.getCommands())
        x += position.x_advance + LETTER_SPACING_EM * upem
    assert bounds.bounds is not None, "the wordmark drew no outline"
    x_min, y_min, x_max, y_max = bounds.bounds
    pad = PADDING_EM * upem
    left, top, width, height = x_min - pad, y_min - pad, x_max - x_min + 2 * pad, y_max - y_min + 2 * pad
    d = " ".join(paths)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{left:.0f} {top:.0f} {width:.0f} {height:.0f}" '
        f'width="{width / upem * 100:.0f}" height="{height / upem * 100:.0f}" role="img" aria-label="{WORDMARK}">\n'
        f"<title>{WORDMARK}</title>\n"
        f'<linearGradient id="brand" gradientUnits="userSpaceOnUse" x1="{x_min:.0f}" y1="0" x2="{x_max:.0f}" y2="0">'
        f'<stop offset="0" stop-color="{CANVAS_RED}"/><stop offset="1" stop-color="{GLIMPSE_BLUE}"/></linearGradient>\n'
        f'<path fill="url(#brand)" d="{d}"/>\n'
        "</svg>\n"
    )


def neurips_on_dark(svg: str) -> str:
    assert svg.count(NEURIPS_TEXT_FILL) == 2, "the NeurIPS logo's text fill changed; check the downloaded file"
    return svg.replace(NEURIPS_TEXT_FILL, NEURIPS_TEXT_FILL_ON_DARK)


def main(args: Args) -> None:
    (LOGOS / "canvit-wordmark.svg").write_text(wordmark_svg(args.font))
    (LOGOS / "neurips-dark.svg").write_text(neurips_on_dark((LOGOS / "neurips.svg").read_text()))


if __name__ == "__main__":
    main(tyro.cli(Args))
