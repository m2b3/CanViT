from pathlib import Path
from typing import Any, Self

from canvit_core import CanViTConfig
from canvit_core.checkpoint import CheckpointMixin
from mlx import nn


class HubMixin(CheckpointMixin):
    backend = "mlx"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        return cls(**(config | {"canvit_config": CanViTConfig(**config["canvit_config"])}))

    def _save_weights(self, path: Path) -> None:
        assert isinstance(self, nn.Module)
        self.save_weights(str(path))

    def _load_weights(self, path: Path) -> None:
        assert isinstance(self, nn.Module)
        self.load_weights(str(path), strict=True)
