"""canvit_nnx.releases names repositories that exist."""

import pytest
from huggingface_hub import HfApi

from canvit_nnx import releases

LISTED = [
    *releases.PRETRAINED.values(), releases.DINOV3_VITB16_IN1K_PROBE, releases.ARCHITECTURE_CONTROL,
    releases.ARCHITECTURE_CONTROL_ADE20K_PROBE, *releases.COCO_DETECTORS.values(),
]


@pytest.mark.network
@pytest.mark.parametrize("repo", LISTED)
def test_listed_release_is_public(repo: str) -> None:
    assert not HfApi().model_info(repo).private
