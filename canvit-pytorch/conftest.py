"""Pytest configuration: matplotlib draws without a display, and tests marked gated(repo_id) are skipped where the
gated Hugging Face repo cannot be read."""

import functools

import matplotlib
import pytest
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

matplotlib.use("Agg")


def pytest_runtest_setup(item: pytest.Item) -> None:
    for marker in item.iter_markers("gated"):
        if (reason := _unreadable(*marker.args)) is not None:
            pytest.skip(reason)


@functools.cache
def _unreadable(repo_id: str) -> str | None:
    """Why this environment cannot read the gated repo_id, or None when it can."""
    try:
        hf_hub_download(repo_id, "config.json")
    except GatedRepoError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        return (f"{repo_id} is gated and this environment cannot read it (HTTP {status}): accept its license at "
                f"https://huggingface.co/{repo_id}, then log in with `hf auth login` or set HF_TOKEN")
    return None
