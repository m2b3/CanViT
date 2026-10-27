import os

# Two CPU devices for the data-parallel tests, set before JAX initializes its backends.
os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=2".strip()

import pytest
from canvit_core import CanViTConfig
from canvit_core.backbone import BACKBONES, ViTSpec


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "network: needs released files from the Hugging Face Hub")


@pytest.fixture
def small_config(monkeypatch: pytest.MonkeyPatch) -> CanViTConfig:
    """A CanViT small enough for CPU tests, with two Canvas Attention Reads and two Writes."""
    monkeypatch.setitem(
        BACKBONES, "vits16", ViTSpec(embed_dim=64, num_heads=4, num_blocks=4, patch_size=4, layerscale_init=0.2)
    )
    return CanViTConfig(
        backbone_name="vits16",
        canvas_num_heads=2,
        canvas_head_dim=16,
        num_canvas_registers=2,
        num_backbone_registers=1,
        rw_stride=1,
    )

