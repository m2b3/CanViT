from pathlib import Path

import pytest

from canvit_pytorch.project import BIBTEX

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("page", ["README.md", "site/index.html"])
def test_citation_matches_project_bibtex(page: str):
    """The README and the project page print the citation that the Hub cards render from `project.BIBTEX`, up to line
    wrapping, which BibTeX reads as a space."""
    assert " ".join(BIBTEX.split()) in " ".join((REPOSITORY / page).read_text().split())
