"""Every number marked data-macro="name", on the page or in the repository README, must read exactly as the
paper's macro of that name.

    python3 site/check_paper_numbers.py

assets/paper/data_macros.json is a copy of the paper's generated macros (see README.md, "Paper numbers").
"""

import json
import re
import sys
from pathlib import Path

SITE = Path(__file__).parent
MARKED = re.compile(r'<span data-macro="(\w+)">([^<]*)</span>')

macros = json.loads((SITE / "assets/paper/data_macros.json").read_text())
errors, checked = [], 0
for source in [*sorted(SITE.rglob("*.html")), SITE.parent / "README.md"]:
    text = source.read_text()
    marks = MARKED.findall(text)
    if len(marks) != text.count("data-macro="):
        errors.append(f"{source}: a data-macro attribute is not on a <span> holding only text")
    for name, shown in marks:
        checked += 1
        if name not in macros:
            errors.append(f"{source}: no paper macro named {name}")
        elif shown != macros[name]:
            errors.append(f"{source}: {name} reads {shown!r}, the paper says {macros[name]!r}")
# ade20k_seg.json (the ADE20K results chart's data) must be the export behind the same macros.
ade20k = json.loads((SITE / "assets/paper/ade20k_seg.json").read_text())
claims = ade20k["claims"]
for macro, value in [("adeBestPriorMiou", claims["best_prior_miou_pct"]), ("adeBestMiou", claims["best_miou_pct"]),
                     ("adeBestPriorGflops", claims["best_prior_gflops"])]:
    checked += 1
    if f"{value:.1f}" != macros[macro]:
        errors.append(f"assets/paper/ade20k_seg.json: {macro} would read {value:.1f}, the paper says {macros[macro]!r}")
if errors:
    sys.exit("\n".join(errors))
print(f"{checked} numbers match the paper's macros")
