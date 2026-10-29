# /// script
# requires-python = ">=3.12"
# dependencies = ["pymupdf>=1.26,<2", "tyro"]
# ///
"""The paper's figures and tables on a talk's slides: tried locally, then accepted into the talk or rejected.

A talk lists the figures and tables it may show in its sources/paper-figures.json, by the paper's own names: a figure
by its export (paper/exports/NAME.pdf), a table by its LaTeX label. Every enabled entry is a slide, which
paper-figures.js (beside this file) inserts after the slide the entry's `where` names, or at the end of the appendix
slides. The deck is public while the paper's next version may not be, so a render reaches a talk in two steps:

- `try` renders entries from a commit of a paper checkout into the talk's .paper-figures/ (ignored by git, never
  published) and enables them; on reload the deck shows each, marked as a trial.
- `accept` moves a trial render into the talk's tracked assets/paper-figures/ and records in its entry the commit,
  source, renderer and checksum; committing the render and the manifest is what publishes it.
- `reject` deletes an entry's renders and resets the entry to how `sync` wrote it, so nothing of the trial remains.

deploy.sh runs `check`, which fails on an enabled entry that is not accepted, an accepted render that differs from its
record, a render in assets/paper-figures/ that no entry accepts, and a `where` that names no slide.

    uv run site/slides/paper_figures.py sync --talk TALK --paper CHECKOUT      # list the paper's figures and tables
    uv run site/slides/paper_figures.py try --talk TALK --paper CHECKOUT ID...
    uv run site/slides/paper_figures.py accept --talk TALK ID...
    uv run site/slides/paper_figures.py reject --talk TALK ID...
    uv run site/slides/paper_figures.py check                                  # every talk; deploy.sh runs it

A figure is its export, converted to SVG by pdftocairo. A table is cut from the camera-ready build of the paper at the
same commit, from its top rule to its bottom rule, so its caption stays off the slide; the build is the paper's own
(generated data checked, camera_ready_wrapper.tex compiled by tectonic) with the drafting marks \\agenttext and
\\postedtext typeset in black.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from datetime import date
from functools import cache
from pathlib import Path
from typing import Annotated, Literal, get_args

import tyro

TALKS = Path(__file__).resolve().parent
"""site/slides/: a talk is a directory here."""
MANIFEST = Path("sources/paper-figures.json")
TRIALS = Path(".paper-figures")
ACCEPTED = Path("assets/paper-figures")
PROVENANCE = "PROVENANCE.md"
"""The one file of ACCEPTED that is not a render: it points to the manifest."""
APPENDIX = "appendix"
"""The `where` of an entry shown at the end of the appendix, and the id of the section that opens it."""

MANUSCRIPT = "paper/latex/CanViT_Toward_AVFMs.tex"
CAMERA_READY = "camera_ready_wrapper"
"""The camera-ready build's wrapper in paper/latex/ (paper/justfile, build-camera-ready)."""
EXPORTS = "paper/exports"
INCLUDED_EXPORT = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{figures/exported/([^}]+)\.pdf\}")
"""How the manuscript includes an export: latex/figures/exported links to paper/exports."""
TALK_BUILD = "talk_build"
TALK_BUILD_TEX = rf"""% The camera-ready build with the drafting marks in black, for talks (site/slides/paper_figures.py).
\AtBeginDocument{{\renewcommand{{\agenttext}}[1]{{#1}}\renewcommand{{\postedtext}}[1]{{#1}}}}
\input{{{CAMERA_READY}}}
"""

ABOUT = (
    "The paper's figures and tables this talk may show, by the paper's names: a figure by its export "
    "(paper/exports/NAME.pdf), a table by its LaTeX label. An enabled entry is a slide after the slide its `where` "
    'names, or at the end of the appendix for "appendix"; `title` (HTML) heads it, `notes` (HTML) are its speaker '
    "notes, `status` is its data-status (draft unless given). `accepted` records the render the talk ships. "
    'site/slides/paper_figures.py writes this file; the talk\'s AGENTS.md, "Paper figures", says how to use it.'
)

Kind = Literal["figure", "table"]


def slug(id: str) -> str:
    """An entry's file name and slide id: its id with ':' (in LaTeX labels) replaced, as paper-figures.js does."""
    return id.replace(":", "-")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shown(path: Path) -> str:
    """A path as messages give it: relative to the working directory (the repository's root, as the usage shows)."""
    return os.path.relpath(path)


def run(command: list[str], cwd: Path | None = None, stdin: bytes | None = None) -> bytes:
    """Run a command; on failure, raise with the command and its output."""
    result = subprocess.run(command, cwd=cwd, input=stdin, capture_output=True, check=False)
    if result.returncode != 0:
        output = (result.stdout + result.stderr).decode(errors="replace").strip().splitlines()
        tail = "\n".join(output[-40:])
        raise SystemExit(f"{' '.join(command)} (in {cwd or Path.cwd()}) exited with {result.returncode}:\n{tail}")
    return result.stdout


@cache
def tool_version(command: tuple[str, ...]) -> str:
    """The first line a tool prints about its version (pdftocairo prints it to stderr)."""
    if shutil.which(command[0]) is None:
        raise SystemExit(f"{command[0]} is not installed; paper_figures.py needs it to render the paper's figures")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout or result.stderr).strip().splitlines()[0]


def pdftocairo_version() -> str:
    return tool_version(("pdftocairo", "-v"))


@dataclass(frozen=True)
class Render:
    """How a render was made, enough to make it again."""

    commit: str
    """The commit of the paper checkout it was rendered from."""
    source: str
    """What was rendered, in the paper's terms."""
    renderer: str
    """The tools and their versions."""
    sha256: str
    """Of the SVG."""


@dataclass(frozen=True)
class Accepted:
    """The render a talk ships for an entry."""

    file: str
    """The render, relative to the talk's directory."""
    sha256: str
    commit: str
    source: str
    renderer: str
    date: str
    """When it was accepted."""


@dataclass(frozen=True)
class Entry:
    """A figure or table of the paper that a talk may show; its defaults are how `sync` writes it."""

    id: str
    """A figure's export name, or a table's LaTeX label."""
    kind: Kind
    enabled: bool = False
    where: str = APPENDIX
    """The id of the slide this one follows, or APPENDIX for the end of the appendix."""
    title: str = ""
    status: str | None = None
    """The slide's data-status; None for draft. deck.js checks the value."""
    notes: tuple[str, ...] = ()
    accepted: Accepted | None = None

    def to_json(self) -> dict:
        data = {"id": self.id, "kind": self.kind, "enabled": self.enabled, "where": self.where, "title": self.title}
        if self.status is not None:
            data["status"] = self.status
        if self.notes:
            data["notes"] = list(self.notes)
        if self.accepted is not None:
            data["accepted"] = asdict(self.accepted)
        return data

    @classmethod
    def from_json(cls, data: dict, where: str) -> "Entry":
        """An entry read from a manifest; `where` names the manifest in errors."""
        known = {f.name for f in fields(cls)}
        if unknown := set(data) - known:
            raise SystemExit(f"{where}: entry {data.get('id')!r} has unknown fields {sorted(unknown)}")
        try:
            entry = cls(
                id=data["id"],
                kind=data["kind"],
                enabled=data["enabled"],
                where=data["where"],
                title=data["title"],
                status=data.get("status"),
                notes=tuple(data.get("notes", ())),
                accepted=Accepted(**data["accepted"]) if "accepted" in data else None,
            )
        except (KeyError, TypeError) as error:
            raise SystemExit(f"{where}: entry {data.get('id')!r} is malformed: {error!r}") from error
        if entry.kind not in get_args(Kind):
            raise SystemExit(f"{where}: entry {entry.id!r} has kind {entry.kind!r}, not one of {get_args(Kind)}")
        if not isinstance(entry.enabled, bool) or not entry.where or not isinstance(entry.title, str):
            raise SystemExit(f"{where}: entry {entry.id!r} needs a boolean `enabled`, a `where` and a string `title`")
        if not isinstance(data.get("notes", []), list) or not all(isinstance(n, str) for n in entry.notes):
            raise SystemExit(f"{where}: entry {entry.id!r} has `notes` that are not a list of strings")
        if entry.status is not None and not isinstance(entry.status, str):
            raise SystemExit(f"{where}: entry {entry.id!r} has a `status` that is not a string")
        return entry


@dataclass(frozen=True)
class Manifest:
    entries: tuple[Entry, ...]

    def __post_init__(self) -> None:
        slugs = [slug(e.id) for e in self.entries]
        if duplicates := {s for s in slugs if slugs.count(s) > 1}:
            raise SystemExit(f"{MANIFEST}: entries share a file name: {sorted(duplicates)}")

    def get(self, id: str) -> Entry:
        for entry in self.entries:
            if entry.id == id:
                return entry
        raise SystemExit(f"{id!r} is not in {MANIFEST}; `sync` adds the paper's figures and tables to it")

    def named(self, ids: tuple[str, ...]) -> list[Entry]:
        if not ids:
            raise SystemExit("name at least one entry: a figure's export name or a table's LaTeX label")
        return [self.get(id) for id in ids]

    def with_entry(self, entry: Entry) -> "Manifest":
        return Manifest(tuple(entry if e.id == entry.id else e for e in self.entries))

    def dumps(self) -> str:
        """One entry per line, so a change to an entry is a one-line diff."""
        lines = [json.dumps(e.to_json(), ensure_ascii=False) for e in self.entries]
        body = ",\n".join(f"  {line}" for line in lines)
        return f'{{\n "_about": {json.dumps(ABOUT, ensure_ascii=False)},\n "figures": [\n{body}\n ]\n}}\n'

    @classmethod
    def loads(cls, text: str, where: str) -> "Manifest":
        data = json.loads(text)
        if set(data) != {"_about", "figures"}:
            raise SystemExit(f"{where}: expected the keys _about and figures, found {sorted(data)}")
        return cls(tuple(Entry.from_json(entry, where) for entry in data["figures"]))


@dataclass(frozen=True)
class Talk:
    dir: Path

    @classmethod
    def named(cls, name: str) -> "Talk":
        talk = cls(TALKS / name)
        if not (talk.dir / "index.html").is_file():
            raise SystemExit(f"{talk.dir} is not a talk: it has no index.html")
        return talk

    @property
    def manifest_path(self) -> Path:
        return self.dir / MANIFEST

    def load(self) -> Manifest:
        if not self.manifest_path.is_file():
            raise SystemExit(f"{self.manifest_path} does not exist; `sync` creates it")
        return Manifest.loads(self.manifest_path.read_text(), str(self.manifest_path))

    def save(self, manifest: Manifest) -> None:
        self.manifest_path.write_text(manifest.dumps())

    def trial(self, id: str, suffix: str) -> Path:
        return self.dir / TRIALS / f"{slug(id)}{suffix}"

    def accepted_file(self, id: str) -> Path:
        return self.dir / ACCEPTED / f"{slug(id)}.svg"

    def slide_ids(self) -> set[str]:
        return set(re.findall(r'<section\b[^>]*\bid="([^"]+)"', (self.dir / "index.html").read_text()))

    def check_where(self, entry: Entry) -> None:
        if entry.where != APPENDIX and entry.where not in self.slide_ids():
            raise SystemExit(f"{entry.id}: where={entry.where!r} names no slide of {shown(self.dir / 'index.html')}")


@dataclass(frozen=True)
class PaperCommit:
    """A commit of a checkout of this repository, whose paper/ the renders come from."""

    checkout: Path
    commit: str

    @classmethod
    def resolve(cls, path: Path, rev: str) -> "PaperCommit":
        checkout = Path(run(["git", "-C", str(path), "rev-parse", "--show-toplevel"]).decode().strip())
        commit = run(["git", "-C", str(checkout), "rev-parse", "--verify", f"{rev}^{{commit}}"]).decode().strip()
        return cls(checkout, commit)

    def read(self, path: str) -> bytes:
        found = subprocess.run(
            ["git", "-C", str(self.checkout), "cat-file", "-e", f"{self.commit}:{path}"], check=False
        )
        if found.returncode != 0:
            raise SystemExit(f"{path} is not in commit {self.commit} of {self.checkout}")
        return run(["git", "-C", str(self.checkout), "show", f"{self.commit}:{path}"])

    def extract(self, into: Path) -> None:
        archive = run(["git", "-C", str(self.checkout), "archive", self.commit, "paper"])
        run(["tar", "-x", "-C", str(into)], stdin=archive)


def figure_svg(paper: PaperCommit, id: str, out: Path) -> Render:
    """The export `id` at the paper's commit, converted to SVG."""
    import pymupdf

    source = f"{EXPORTS}/{id}.pdf"
    with tempfile.TemporaryDirectory(prefix="paper-figure-") as tmp:
        pdf = Path(tmp) / f"{slug(id)}.pdf"
        pdf.write_bytes(paper.read(source))
        if (pages := pymupdf.open(pdf).page_count) != 1:
            raise SystemExit(f"{source} at {paper.commit} has {pages} pages; a figure export has one")
        run(["pdftocairo", "-svg", str(pdf), str(out)])
    return Render(paper.commit, source, pdftocairo_version(), sha256(out))


def brace_groups(text: str, start: int) -> list[str]:
    """The contents of the consecutive brace groups of TeX source that begin at text[start]."""
    groups, i = [], start
    while i < len(text) and text[i] == "{":
        depth, j = 0, i
        while True:
            if text[j] == "\\":
                j += 2
                continue
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                break
            j += 1
        groups.append(text[i + 1 : j])
        i = j + 1
    return groups


def label_anchors(aux: str) -> dict[str, str]:
    """Each label of a LaTeX build and its hyperref anchor (table.N, figure.N, section.N, ...), in document order:
    hyperref writes \\newlabel{LABEL}{{number}{page}{title}{anchor}{}} to the .aux file."""
    anchors = {}
    for match in re.finditer(r"\\newlabel(?=\{)", aux):
        label, value = brace_groups(aux, match.end())[:2]
        parts = brace_groups(value, 0)
        if len(parts) >= 4 and parts[3]:
            anchors[label] = parts[3]
    return anchors


@dataclass(frozen=True)
class CameraReadyBuild:
    """The paper's camera-ready build at a commit, with the drafting marks in black (TALK_BUILD_TEX)."""

    paper: PaperCommit
    pdf: Path
    anchors: dict[str, str]
    renderer: str

    def table_labels(self) -> list[str]:
        return [label for label, anchor in self.anchors.items() if anchor.startswith("table.")]

    def table_svg(self, label: str, out: Path) -> Render:
        """The table `label`, from its top rule to its bottom rule (booktabs), as SVG."""
        import pymupdf

        anchor = self.anchors.get(label)
        if anchor is None or not anchor.startswith("table."):
            raise SystemExit(f"{label} is not the label of a table in {CAMERA_READY}.tex at {self.paper.commit}")
        document = pymupdf.open(self.pdf)
        page_number, rect = table_rect(document, anchor)
        with tempfile.TemporaryDirectory(prefix="paper-table-") as tmp:
            cut = pymupdf.open()
            cut.insert_pdf(document, from_page=page_number, to_page=page_number)
            page = cut[0]
            # pdftocairo's SVG output spans the MediaBox, which is in PDF coordinates (y up).
            page.set_mediabox((rect * ~page.transformation_matrix).normalize())
            pdf = Path(tmp) / "table.pdf"
            cut.save(pdf, garbage=4, deflate=True)
            run(["pdftocairo", "-svg", str(pdf), str(out)])
        source = f"table {label} of paper/latex/{CAMERA_READY}.tex, built with \\agenttext and \\postedtext in black"
        return Render(self.paper.commit, source, self.renderer, sha256(out))


@contextmanager
def camera_ready_build(paper: PaperCommit) -> Iterator[CameraReadyBuild]:
    """Build the paper at the commit as `just build-camera-ready` does, in a temporary copy of its paper/; the commit's
    generated data must be current, so the build shows the commit as it is."""
    import pymupdf

    renderer = ", ".join(
        [pdftocairo_version(), f"PyMuPDF {pymupdf.VersionBind}", tool_version(("tectonic", "--version"))]
    )
    with tempfile.TemporaryDirectory(prefix="paper-build-") as tmp:
        paper.extract(Path(tmp))
        root = Path(tmp) / "paper"
        run([sys.executable, "latex/generate_data.py", "--check"], cwd=root)
        latex = root / "latex"
        (latex / f"{TALK_BUILD}.tex").write_text(TALK_BUILD_TEX)
        print(f"building paper/latex/{CAMERA_READY}.tex at {paper.commit[:12]} with tectonic", file=sys.stderr)
        run(["tectonic", "--reruns", "3", "--keep-intermediates", f"{TALK_BUILD}.tex"], cwd=latex)
        anchors = label_anchors((latex / f"{TALK_BUILD}.aux").read_text())
        yield CameraReadyBuild(paper, latex / f"{TALK_BUILD}.pdf", anchors, renderer)


@dataclass(frozen=True)
class Rule:
    """A horizontal line on a page: y and x in points from the top left, thickness its stroke or height."""

    y: float
    x0: float
    x1: float
    thickness: float


def horizontal_rules(page) -> list[Rule]:
    """The horizontal lines and thin filled rectangles at least 20 pt long (booktabs rules are either)."""
    rules = []
    for path in page.get_drawings():
        for item in path["items"]:
            if item[0] == "l" and "s" in path["type"]:
                start, end = item[1], item[2]
                if abs(start.y - end.y) < 0.01 and abs(end.x - start.x) >= 20:
                    rules.append(Rule(start.y, min(start.x, end.x), max(start.x, end.x), path["width"] or 0.4))
            elif item[0] == "re" and "f" in path["type"]:
                rect = item[1]
                if rect.height <= 1.5 and rect.width >= 20:
                    rules.append(Rule((rect.y0 + rect.y1) / 2, rect.x0, rect.x1, rect.height))
    return rules


FOLLOWING_ANCHOR = re.compile(r"(table|figure|section|subsection|subsubsection|paragraph|appendix)\.")
"""Anchors that mark where a table's float has ended, when one follows it on its page."""


def table_rect(document, anchor: str):
    """The page and the rectangle (points, from the top left) from a table's top rule to its bottom rule.

    The table's caption, above it, holds its anchor; its top rules are the first rules below the anchor (side by side
    for tables set in minipages), and each one's bottom rule is the last rule of the same span before the next anchor
    on the page."""
    import pymupdf

    destinations = document.resolve_names()
    page_number = destinations[anchor]["page"]
    page = document[page_number]

    def top_left(name: str):
        """A destination's point in page coordinates (from the top left); PyMuPDF gives it in PDF coordinates."""
        return pymupdf.Point(destinations[name]["to"]) * page.transformation_matrix

    anchor_y = top_left(anchor).y
    following = [
        top_left(name).y
        for name, d in destinations.items()
        if d["page"] == page_number and FOLLOWING_ANCHOR.match(name) and top_left(name).y > anchor_y + 1
    ]
    limit = min(following, default=page.rect.height)
    below = [r for r in horizontal_rules(page) if anchor_y < r.y < limit]
    if not below:
        raise SystemExit(f"no rule below the caption of {anchor} on page {page_number + 1}: is it a booktabs table?")
    first = min(r.y for r in below)
    rect = pymupdf.Rect()
    for top in (r for r in below if r.y - first < 2):
        same_span = [r for r in below if abs(r.x0 - top.x0) < 0.6 and abs(r.x1 - top.x1) < 0.6]
        bottom = max(same_span, key=lambda r: r.y)
        rect |= pymupdf.Rect(top.x0, top.y - top.thickness / 2, top.x1, bottom.y + bottom.thickness / 2)
    return page_number, rect + (-2, -2, 2, 2)


def render(paper: PaperCommit, entries: list[Entry], out: Callable[[str], Path]) -> dict[str, Render]:
    """Render entries at the paper's commit to out(id), building the paper once if any is a table."""
    renders = {e.id: figure_svg(paper, e.id, out(e.id)) for e in entries if e.kind == "figure"}
    if tables := [e for e in entries if e.kind == "table"]:
        with camera_ready_build(paper) as build:
            renders |= {e.id: build.table_svg(e.id, out(e.id)) for e in tables}
    return renders


def paper_order(paper: PaperCommit) -> list[Entry]:
    """The paper's figures (exports the manuscript includes) and tables (labels of table floats), in its order."""
    manuscript = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in paper.read(MANUSCRIPT).decode().splitlines())
    positioned = [(m.start(), Entry(m.group(1), "figure")) for m in INCLUDED_EXPORT.finditer(manuscript)]
    with camera_ready_build(paper) as build:
        labels = build.table_labels()
    for label in labels:
        position = manuscript.find(f"\\label{{{label}}}")
        positioned.append((position if position >= 0 else len(manuscript), Entry(label, "table")))
    entries = [entry for _, entry in sorted(positioned, key=lambda pair: pair[0])]
    assert len({e.id for e in entries}) == len(entries), f"an export or label appears twice in {MANUSCRIPT}"
    return entries


Ids = Annotated[tuple[str, ...], tyro.conf.Positional]


@dataclass(frozen=True)
class Sync:
    """Add the paper's figures and tables at a commit to the talk's manifest, disabled and in the appendix, in the paper's
    order; entries already there keep their settings, and entries the paper no longer has are reported and kept."""

    talk: str
    """The talk's directory under site/slides/."""
    paper: Path
    """A checkout of this repository holding the paper version to read."""
    rev: str = "HEAD"
    """The commit of the checkout to read (a commit, branch or tag)."""

    def run(self) -> None:
        talk = Talk.named(self.talk)
        current = talk.load().entries if talk.manifest_path.exists() else ()
        known = {e.id: e for e in current}
        paper = PaperCommit.resolve(self.paper, self.rev)
        ordered = paper_order(paper)
        in_paper = {e.id for e in ordered}
        gone = [e for e in current if e.id not in in_paper]
        added = [e for e in ordered if e.id not in known]
        for entry in gone:
            print(f"{entry.id}: not in the paper at {paper.commit[:12]}; kept at the end of the manifest")
        talk.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        talk.save(Manifest(tuple(known.get(e.id, e) for e in ordered) + tuple(gone)))
        print(f"{shown(talk.manifest_path)}: {len(added)} added ({', '.join(e.id for e in added) or 'none'})")


@dataclass(frozen=True)
class Try:
    """Render entries from the paper at a commit into the talk's .paper-figures/ and enable their slides. An accepted
    entry goes back on trial: its accepted render stays in place until `accept` replaces it or `reject` deletes it."""

    ids: Ids
    """Entries by id: a figure's export name or a table's LaTeX label."""
    talk: str
    """The talk's directory under site/slides/."""
    paper: Path
    """A checkout of this repository holding the paper version to render."""
    rev: str = "HEAD"
    """The commit of the checkout to read (a commit, branch or tag)."""
    where: str | None = None
    """Place the slides after this slide (its id), or at the end of the appendix ("appendix")."""

    def run(self) -> None:
        talk = Talk.named(self.talk)
        manifest = talk.load()
        entries = manifest.named(self.ids)
        if self.where is not None:
            entries = [replace(e, where=self.where) for e in entries]
        for entry in entries:
            talk.check_where(entry)
        paper = PaperCommit.resolve(self.paper, self.rev)
        (talk.dir / TRIALS).mkdir(exist_ok=True)
        renders = render(paper, entries, lambda id: talk.trial(id, ".svg"))
        for entry in entries:
            talk.trial(entry.id, ".json").write_text(json.dumps(asdict(renders[entry.id]), indent=1) + "\n")
            if entry.accepted is not None:
                print(
                    f"{entry.id}: back on trial; {entry.accepted.file} stays until `accept` replaces it or `reject` "
                    f"deletes it (`git checkout {MANIFEST}` keeps it instead)"
                )
            manifest = manifest.with_entry(replace(entry, enabled=True, accepted=None))
            place = "at the end of the appendix" if entry.where == APPENDIX else f"after #{entry.where}"
            print(
                f"{entry.id}: on trial {place}, rendered from {paper.commit[:12]} to {shown(talk.trial(entry.id, '.svg'))}"
            )
        talk.save(manifest)


@dataclass(frozen=True)
class Accept:
    """Move entries' trial renders into the talk's assets/paper-figures/ and record their provenance in the manifest.
    The talk's branch is public: once committed and pushed, the renders are."""

    ids: Ids
    """Entries by id: a figure's export name or a table's LaTeX label."""
    talk: str
    """The talk's directory under site/slides/."""

    def run(self) -> None:
        talk = Talk.named(self.talk)
        manifest = talk.load()
        (talk.dir / ACCEPTED).mkdir(parents=True, exist_ok=True)
        for entry in manifest.named(self.ids):
            id = entry.id
            svg, record = talk.trial(id, ".svg"), talk.trial(id, ".json")
            if not (svg.is_file() and record.is_file()):
                raise SystemExit(f"{id}: no trial render in {shown(talk.dir / TRIALS)}; `try` makes one")
            made = Render(**json.loads(record.read_text()))
            if sha256(svg) != made.sha256:
                raise SystemExit(f"{id}: {shown(svg)} changed since `try` rendered it; render it again")
            target = talk.accepted_file(id)
            shutil.move(svg, target)
            record.unlink()
            file = str(target.relative_to(talk.dir))
            accepted = Accepted(file, made.sha256, made.commit, made.source, made.renderer, date.today().isoformat())
            manifest = manifest.with_entry(replace(entry, accepted=accepted))
            print(f"{id}: accepted as {shown(target)} (commit {made.commit[:12]})")
        talk.save(manifest)
        print(f"to keep them: git add {shown(talk.manifest_path)} {shown(talk.dir / ACCEPTED)} and commit")


@dataclass(frozen=True)
class Reject:
    """Delete entries' renders, trial and accepted, and reset their entries as `sync` writes them."""

    ids: Ids
    """Entries by id: a figure's export name or a table's LaTeX label."""
    talk: str
    """The talk's directory under site/slides/."""

    def run(self) -> None:
        talk = Talk.named(self.talk)
        manifest = talk.load()
        for entry in manifest.named(self.ids):
            id = entry.id
            removed = [
                p for p in (talk.trial(id, ".svg"), talk.trial(id, ".json"), talk.accepted_file(id)) if p.exists()
            ]
            for path in removed:
                path.unlink()
            reset = Entry(entry.id, entry.kind)
            manifest = manifest.with_entry(reset)
            was = "" if entry == reset else f"; its entry was {json.dumps(entry.to_json(), ensure_ascii=False)}"
            print(f"{id}: rejected, removed {', '.join(map(shown, removed)) or 'no render'}{was}")
        talk.save(manifest)
        trials = talk.dir / TRIALS
        if trials.is_dir() and not any(trials.iterdir()):
            trials.rmdir()


def problems(talk: Talk, manifest: Manifest) -> list[str]:
    """What keeps a talk's paper figures from being published as they stand."""
    found = []
    slides = talk.slide_ids()
    accepted_files = set()
    for entry in manifest.entries:
        if entry.where != APPENDIX and entry.where not in slides:
            found.append(f"{entry.id}: where={entry.where!r} names no slide")
        if entry.enabled and entry.accepted is None:
            found.append(f"{entry.id}: enabled but not accepted (`accept` it, or disable or `reject` it)")
        if entry.accepted is not None:
            file = talk.dir / entry.accepted.file
            accepted_files.add(file)
            if file != talk.accepted_file(entry.id):
                found.append(
                    f"{entry.id}: accepted file {entry.accepted.file} is not {shown(talk.accepted_file(entry.id))}"
                )
            elif not file.is_file():
                found.append(f"{entry.id}: accepted file {shown(file)} is missing")
            elif sha256(file) != entry.accepted.sha256:
                found.append(f"{entry.id}: {shown(file)} differs from the render accepted on {entry.accepted.date}")
    if (talk.dir / ACCEPTED).is_dir():
        for file in sorted((talk.dir / ACCEPTED).iterdir()):
            if file.name != PROVENANCE and file not in accepted_files:
                found.append(f"{shown(file)}: no entry accepts it (`accept` its entry again, or `reject` it)")
    return found


@dataclass(frozen=True)
class Check:
    """Check every talk's paper figures for publication: deploy.sh runs this and stops on a problem."""

    def run(self) -> None:
        talks = sorted(path.parent.parent for path in TALKS.glob(f"*/{MANIFEST}"))
        failed = False
        for directory in talks:
            talk = Talk(directory)
            manifest = talk.load()
            found = problems(talk, manifest)
            shown = sum(e.enabled for e in manifest.entries)
            accepted = sum(e.accepted is not None for e in manifest.entries)
            print(f"{directory.name}: {shown} enabled, {accepted} accepted, of {len(manifest.entries)} paper figures")
            for problem in found:
                print(f"  {problem}")
            failed |= bool(found)
        if failed:
            raise SystemExit("paper figures: fix the problems above before publishing")


Command = (
    Annotated[Sync, tyro.conf.subcommand("sync")]
    | Annotated[Try, tyro.conf.subcommand("try")]
    | Annotated[Accept, tyro.conf.subcommand("accept")]
    | Annotated[Reject, tyro.conf.subcommand("reject")]
    | Annotated[Check, tyro.conf.subcommand("check")]
)

if __name__ == "__main__":
    tyro.cli(Command, description=__doc__).run()
