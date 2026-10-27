from dataclasses import dataclass
from pathlib import Path


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("canvit_paper_exporter must live inside a repo with a pyproject.toml")


REPO_ROOT = _find_repo_root()
PAPER_ROOT = REPO_ROOT.parent
DATA      = REPO_ROOT / "data"
# The manuscript's inputs: paper/latex/figures/exported links here.
EXPORTS   = PAPER_ROOT / "exports"


@dataclass(frozen=True)
class Inputs:
    """The files of a data/ entry that a dataset or figure reads: those matching its glob patterns.

    Datasets and figures reach data/ only through the Inputs they declare, so the declarations list what they read.
    """
    entry: str
    patterns: tuple[str, ...]

    @property
    def directory(self) -> Path:
        return DATA / self.entry

    def files(self) -> list[Path]:
        """Every matching file, in path order; each pattern must match a file."""
        found: set[Path] = set()
        for pattern in self.patterns:
            matches = {f for f in self.directory.glob(pattern) if f.is_file()}
            assert matches, f"No files matching {pattern!r} in {self.directory}"
            found |= matches
        return sorted(found)

    def file(self, name: str) -> Path:
        """A file this declares by name, relative to the entry."""
        assert name in self.patterns, f"{name!r} is not declared in {self}"
        path = self.directory / name
        assert path.is_file(), f"Missing {path}"
        return path


def export_json(dataset: str) -> Path:
    """Path of a dataset's output JSON."""
    return EXPORTS / f"{dataset}.json"


def export_svg(figure: str) -> Path:
    """Path of a figure's output SVG."""
    return EXPORTS / f"{figure}.svg"


def export_pdf(figure: str) -> Path:
    """Path of a figure's output PDF (native vector — Typst uses SVG, LaTeX uses PDF)."""
    return EXPORTS / f"{figure}.pdf"


def export_png(figure: str) -> Path:
    """Path of a figure's preview PNG."""
    return EXPORTS / f"{figure}.png"
