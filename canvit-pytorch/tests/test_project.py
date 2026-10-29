from pathlib import Path

import pytest

from canvit_pytorch.project import BIBTEX

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("page", ["README.md", "site/index.html"])
def test_citation_matches_project_bibtex(page: str):
    """The README and the project page print the citation that the Hub cards render from `project.BIBTEX`."""
    assert BIBTEX in (REPOSITORY / page).read_text()
