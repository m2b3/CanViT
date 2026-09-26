"""Every page number marked data-macro="name" must read exactly as the paper's macro of that name.

    python3 site/check_paper_numbers.py

assets/paper/data_macros.json is a copy of the paper's generated macros (see README.md, "Paper figures").
"""

import json
import re
import sys
from pathlib import Path

SITE = Path(__file__).parent
MARKED = re.compile(r'<span data-macro="(\w+)">([^<]*)</span>')

macros = json.loads((SITE / "assets/paper/data_macros.json").read_text())
errors, checked = [], 0
for page in sorted(SITE.rglob("*.html")):
    html = page.read_text()
    marks = MARKED.findall(html)
    if len(marks) != html.count("data-macro="):
        errors.append(f"{page}: a data-macro attribute is not on a <span> holding only text")
    for name, text in marks:
        checked += 1
        if name not in macros:
            errors.append(f"{page}: no paper macro named {name}")
        elif text != macros[name]:
            errors.append(f"{page}: {name} reads {text!r}, the paper says {macros[name]!r}")
if errors:
    sys.exit("\n".join(errors))
print(f"{checked} marked numbers match the paper's macros")
