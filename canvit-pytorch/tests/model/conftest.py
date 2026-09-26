import pytest
import torch

from canvit_pytorch import Viewpoint
from tests.model.tiny import glimpse_batch


@pytest.fixture
def glimpses() -> tuple[torch.Tensor, Viewpoint]:
    return glimpse_batch(seed=0, batch_size=2)
