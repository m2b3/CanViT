# /// script
# requires-python = ">=3.12"
# dependencies = ["tyro"]
# ///
"""A talk's files for an organizer's shared folder, in one folder under one name: the deck as one self-contained HTML
file (single_file.py), and every click as a PDF page and as a PowerPoint page (pdf_backup.py).

    node site/serve.mjs 8765 &                            # from the repository root
    uv run site/slides/export_talk.py --talk 2026-10-29-Montreal-AI-and-Neuroscience-MAIN \
        --name 2026-10-29_MAIN2026_Toward-AVFMs_Berreby-Du

Writes site/.screens/export/NAME/ with NAME_interactive.html, NAME_static.pdf and NAME_static.pptx. With --no-build,
takes the files the two builders last wrote. Refuses a build whose own check reported a problem.
"""

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

SCREENS = Path(__file__).resolve().parent.parent / ".screens"


@dataclass(frozen=True)
class Export:
    talk: str
    """the talk's directory under site/slides/"""
    name: str
    """the files' common name: date, venue, talk, authors"""
    build: bool = True
    """rebuild the PDF, PowerPoint and HTML first (about 12 minutes; the dev server must run)"""
    local: str = "http://127.0.0.1:8765"


def run(*command: str) -> None:
    if subprocess.run(["uv", "run", *command]).returncode:
        raise SystemExit(f"export: {command[0]} failed")


def main(args: Export) -> int:
    pdf_dir, single_dir = SCREENS / "pdf" / args.talk, SCREENS / "single"
    if args.build:
        run(str(Path(__file__).with_name("pdf_backup.py")), "--url", f"{args.local}/slides/{args.talk}/", "--out", str(pdf_dir))
        run(str(Path(__file__).with_name("single_file.py")), "--talk", args.talk, "--out", str(single_dir), "--local", args.local)
    for check in (pdf_dir / "check.txt", single_dir / "check.txt"):
        lines = check.read_text().splitlines() if check.exists() else ["missing"]
        if check.parent == pdf_dir and any(l.startswith(("no page", "blank frame", "shoot.py exited")) for l in lines):
            raise SystemExit(f"export: {check} reports a problem")
        if check.parent == single_dir and "no errors" not in " ".join(lines):
            raise SystemExit(f"export: {check} reports a problem")
    out = SCREENS / "export" / args.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for source, suffix in [(single_dir / f"{args.talk}.html", "interactive.html"), (pdf_dir / "slides.pdf", "static.pdf"),
                           (pdf_dir / "slides.pptx", "static.pptx")]:
        shutil.copy2(source, out / f"{args.name}_{suffix}")
    for file in sorted(out.iterdir()):
        print(f"{file}  {file.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Export)))
