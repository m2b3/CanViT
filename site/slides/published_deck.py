"""A talk's deck as published: its Backup slides cut. A talk's Backup is every section from <section id="backup"> to the
end of its slides; the published deck has none of it. deploy.sh rewrites the staged decks with this; check_published.py
serves the local deck through it, so it requests exactly the files the published deck loads.

    python3 site/slides/published_deck.py STAGED/slides/*/index.html   # rewrite each deck in place
"""

import sys
from pathlib import Path

BACKUP = '<section id="backup"'
SLIDES_END = "</div>\n</div>"  # the close of .slides, then of .reveal


def without_backup(html: str) -> str:
    if BACKUP not in html:
        return html
    start = html.index(BACKUP)
    return html[:start] + html[html.index(SLIDES_END, start):]


if __name__ == "__main__":
    for name in sys.argv[1:]:
        path = Path(name)
        path.write_text(without_backup(path.read_text()))
