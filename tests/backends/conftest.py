import importlib
import sys
from pathlib import Path

import pytest
from canvit_core import CanViTConfig

ROOT = Path(__file__).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.backends.support import Backend  # noqa: E402


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--backend", choices=("mlx", "nnx"), default=None)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "network: download released checkpoints from the Hub")


@pytest.fixture(scope="session")
def backend(request: pytest.FixtureRequest) -> Backend:
    name = request.config.getoption("backend")
    if name is None:
        pytest.fail("tests/backends requires --backend=mlx or --backend=nnx")
    module = importlib.import_module(f"canvit_{name}")
    return Backend(name=name, module=module)


@pytest.fixture
def tiny_config(monkeypatch: pytest.MonkeyPatch) -> CanViTConfig:
    from canvit_core.backbone import BACKBONES, ViTSpec

    monkeypatch.setitem(
        BACKBONES,
        "vits16",
        ViTSpec(embed_dim=32, num_heads=4, num_blocks=2, patch_size=4, layerscale_init=0.2),
    )
    return CanViTConfig(
        backbone_name="vits16",
        canvas_num_heads=2,
        canvas_head_dim=8,
        num_canvas_registers=1,
        num_backbone_registers=1,
        rw_stride=1,
    )


@pytest.fixture
def reference_pair(backend: Backend, tiny_config):
    return backend.reference_pair(tiny_config)
