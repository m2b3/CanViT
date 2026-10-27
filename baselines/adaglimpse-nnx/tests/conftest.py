import types

import pytest
import reference
import torch
from torch import nn

from adaglimpse_nnx import ARCHITECTURE_CONTROL


@pytest.fixture(scope="session")
def upstream(request: pytest.FixtureRequest) -> types.SimpleNamespace:
    return reference.import_modules(reference.checkout(request.config.cache.mkdir("upstream")))


@pytest.fixture(scope="session")
def upstream_model(upstream: types.SimpleNamespace) -> nn.Module:
    """A freshly initialized upstream model at the architecture control's configuration, built by the factory
    AdaGlimpse's training harness calls."""
    config = ARCHITECTURE_CONTROL
    patch_area = config.patch_size_px**2
    assert config.teacher_dim % patch_area == 0, "upstream sizes decoder_pred in pixel channels of a patch"
    torch.manual_seed(0)
    return upstream.mae.mae_vit_base_patch16(
        img_size=config.scene_size_px, out_chans=config.teacher_dim // patch_area, num_classes=config.teacher_dim,
    ).eval()
