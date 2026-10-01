"""Writes assets/qr/<name>.svg, a QR code for each of the project's links, read from canvit_pytorch.project.

    uv run --project canvit-pytorch --with segno python site/make_qr_codes.py
"""

from pathlib import Path

import segno

from canvit_pytorch import project

OUT = Path(__file__).parent / "assets" / "qr"
INK = "#0f172a"  # css/canvit.css: --ink
LINKS = {"paper": project.PAPER_URL, "code": project.CODE_URL, "models": project.HUB_ORG_URL, "page": project.PAGE_URL}

OUT.mkdir(exist_ok=True)
for name, url in LINKS.items():
    path = OUT / f"{name}.svg"
    segno.make(url, error="m").save(path, kind="svg", scale=10, border=0, dark=INK, light=None, xmldecl=False,
                                    svgns=True, omitsize=True)
    print(f"{path}  {url}", flush=True)
