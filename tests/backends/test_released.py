from pathlib import Path
from typing import Literal

import pytest
from canvit_pytorch.hub.repos import (
    FINETUNED_IN1K,
    FLAGSHIP,
    RELEASED_CANVAS_GRID_SIZE,
    RELEASED_GLIMPSE_SIZE_PX,
    released_ade20k_probe,
)

from tests.backends.support import Backend
from tools.convert_checkpoints import ConvertCheckpoints, run


@pytest.mark.network
@pytest.mark.parametrize(
    ("source", "kind"),
    [
        (FLAGSHIP, "pretraining"),
        (FINETUNED_IN1K, "classification"),
        (released_ade20k_probe("in21k", scene_size_px=512, canvas_grid_size=64), "probe"),
    ],
)
def test_released_checkpoint_semantics_match_pytorch(
    backend: Backend, source: str, kind: Literal["pretraining", "classification", "probe"], tmp_path: Path,
):
    run(ConvertCheckpoints(
        source=source,
        output=tmp_path / f"{backend.name}-{kind}",
        backend=backend.name,
        kind=kind,
        canvas_grid_size=RELEASED_CANVAS_GRID_SIZE,
        glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX,
        steps=3,
        image=(Path(__file__).parents[2] / "canvit-pytorch" / "test_data" / "Cat03.jpg")
        if kind != "probe" else None,
    ))
