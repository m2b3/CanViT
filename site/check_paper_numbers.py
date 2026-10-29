"""Every number marked data-macro="name", in the site's pages and talks or in the repository README, must read exactly
as the paper's macro of that name.

    uv run --no-project python site/check_paper_numbers.py

assets/paper/data_macros.json links to the paper's generated macros (see README.md, "Paper numbers").
"""

import json
import re
import sys
from pathlib import Path

SITE = Path(__file__).parent
MARKED = re.compile(r'<span data-macro="(\w+)">([^<]*)</span>')
PERCENT_MATH = re.compile(r"^\$(-?)([\d.]+)\\%\$$")


def as_text(value: str) -> str:
    """A macro's value as typeset text. Plain values read as they are; a percentage in math mode, such as the
    ablations' $-98.8\\%$, reads with a typographic minus: −98.8%. Other math has no text form here, which fails
    loudly where a page uses it."""
    if "$" not in value:
        return value
    match = PERCENT_MATH.match(value)
    if not match:
        raise ValueError(f"no text rendering for macro value {value!r}")
    sign, number = match.groups()
    return f"{'−' if sign else ''}{number}%"


macros = json.loads((SITE / "assets/paper/data_macros.json").read_text())
errors, checked = [], 0
pages = [p for p in sorted(SITE.rglob("*.html")) if "node_modules" not in p.parts]
for source in [*pages, SITE.parent / "README.md"]:
    text = source.read_text()
    marks = MARKED.findall(text)
    if len(marks) != text.count("data-macro="):
        errors.append(f"{source}: a data-macro attribute is not on a <span> holding only text")
    for name, shown in marks:
        checked += 1
        if name not in macros:
            errors.append(f"{source}: no paper macro named {name}")
        elif shown != as_text(macros[name]):
            errors.append(f"{source}: {name} reads {shown!r}, the paper says {as_text(macros[name])!r}")
# ade20k_seg.json (the ADE20K results chart's data) must be the export behind the same macros.
ade20k = json.loads((SITE / "assets/paper/ade20k_seg.json").read_text())
claims = ade20k["claims"]
for macro, value in [("adeBestPriorMiou", claims["best_prior_miou_pct"]), ("adeBestMiou", claims["best_miou_pct"]),
                     ("adeBestPriorGflops", claims["best_prior_gflops"])]:
    checked += 1
    if f"{value:.1f}" != macros[macro]:
        errors.append(f"assets/paper/ade20k_seg.json: {macro} would read {value:.1f}, the paper says {macros[macro]!r}")
# A talk's ImageNet-1k references for prior active models must read as the paper's Table 1 prints them, in the row that
# cites the model.
paper_tex = (SITE.parent / "paper/latex/CanViT_Toward_AVFMs.tex").read_text()
for references in sorted(SITE.glob("slides/*/sources/in1k-active-references.json")):
    for ref in json.loads(references.read_text())["references"]:
        checked += 1
        rows = [line for line in paper_tex.splitlines() if f"{ref['name']} \\cite{{{ref['cite']}}}" in line]
        if len(rows) != 1:
            errors.append(f"{references}: {len(rows)} rows of the paper cite {ref['name']} as {ref['cite']}, not one")
        elif (printed := rows[0].split("&")[-1].strip().removesuffix("\\\\").strip()) != f"{ref['top1']:.1f}":
            errors.append(f"{references}: {ref['name']} reads {ref['top1']:.1f}, the paper's Table 1 says {printed!r}")
if errors:
    sys.exit("\n".join(errors))
print(f"{checked} numbers match the paper")
