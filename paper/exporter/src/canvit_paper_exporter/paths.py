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


def eval_dir(dataset: str) -> Path:
    """Directory holding one dataset's raw eval artifacts."""
    return DATA / dataset


def hf_snapshot(name: str) -> Path:
    """One captured HuggingFace config.json."""
    return DATA / "hf" / f"{name}.json"


def comet_curves() -> Path:
    """Directory of captured Comet training-curve JSONs."""
    return DATA / "comet"


def export_json(dataset: str) -> Path:
    """Path of a dataset's output JSON."""
    return EXPORTS / f"{dataset}.json"


def export_svg(figure: str) -> Path:
    """Path of a figure's output SVG."""
    return EXPORTS / f"{figure}.svg"


def export_pdf(figure: str) -> Path:
    """Path of a figure's output PDF (native vector — Typst uses SVG, LaTeX uses PDF)."""
    return EXPORTS / f"{figure}.pdf"
