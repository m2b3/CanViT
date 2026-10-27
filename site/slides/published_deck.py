"""A talk's deck as published: its Backup slides cut. A talk's Backup is every section from <section id="backup"> to the
end of its slides; the published deck has none of it. deploy.sh rewrites the staged decks with this; check_published.py
serves the local deck through it, so it requests exactly the files the published deck loads.

    python3 site/slides/published_deck.py STAGED/slides/*/index.html   # rewrite each deck in place

A deck whose script looks up a Backup slide directly (querySelector("#id")) would fail to start once that slide is
cut, so the rewrite refuses one: a script reaches a Backup slide only through a guard that skips an absent slide
(drawIfPresent in the MAIN deck).
"""

import re
import sys
from pathlib import Path

BACKUP = '<section id="backup"'
SLIDES_END = "</div>\n</div>"  # the close of .slides, then of .reveal


def backup_lookups(html: str) -> list[str]:
    """The Backup slides the deck's page looks up by id outside the slides, which the cut would leave pointing at nothing."""
    if BACKUP not in html:
        return []
    start = html.index(BACKUP)
    end = html.index(SLIDES_END, start)
    cut = set(re.findall(r'<section id="([^"]+)"', html[start:end]))
    looked_up = re.findall(r"""querySelector(?:All)?\(\s*["'`]#([\w-]+)""", html[end:])
    return sorted(set(looked_up) & cut)


def without_backup(html: str) -> str:
    if BACKUP not in html:
        return html
    start = html.index(BACKUP)
    return html[:start] + html[html.index(SLIDES_END, start):]


if __name__ == "__main__":
    for name in sys.argv[1:]:
        path = Path(name)
        html = path.read_text()
        if lookups := backup_lookups(html):
            sys.exit(f"{name}: its script looks up Backup slides directly ({', '.join('#' + i for i in lookups)}); "
                     "reach them through a guard that skips an absent slide")
        path.write_text(without_backup(html))
