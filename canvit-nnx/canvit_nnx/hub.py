"""Native NNX checkpoint loading through the shared CanViT format."""

from pathlib import Path
from typing import Any, ClassVar, Self, cast

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
from canvit_core.checkpoint import CheckpointMixin
from canvit_core.config import CanViTConfig
from safetensors import safe_open
from safetensors.numpy import save_file


def _state_entries(module: nnx.Module) -> list[tuple[tuple[str | int, ...], Any]]:
    state = nnx.to_flat_state(nnx.state(module))
    entries = list(zip(state.paths, state.leaves, strict=True))
    return cast(list[tuple[tuple[str | int, ...], Any]], entries)


def _state_key(path: tuple[str | int, ...]) -> str:
    return ".".join(str(part) for part in path)


class HubMixin(CheckpointMixin):
    backend: ClassVar[str] = "nnx"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        fields = dict(config)
        try:
            encoded_config = fields.pop("canvit_config")
        except KeyError as error:
            raise ValueError("checkpoint config lacks canvit_config") from error
        if not isinstance(encoded_config, dict):
            raise TypeError(f"canvit_config must be an object, got {type(encoded_config).__name__}")
        return cls(**(fields | {"canvit_config": CanViTConfig(**encoded_config), "rngs": nnx.Rngs(0)}))

    def _save_weights(self, path: Path) -> None:
        tensors: dict[str, np.ndarray] = {}
        assert isinstance(self, nnx.Module)
        for state_path, variable in _state_entries(self):
            key = _state_key(state_path)
            if key in tensors:
                raise ValueError(f"duplicate NNX state key {key}")
            tensors[key] = np.asarray(variable[...])
        save_file(tensors, str(path))

    def _load_weights(self, path: Path) -> None:
        assert isinstance(self, nnx.Module)
        entries = _state_entries(self)
        model_by_key = {_state_key(state_path): state_path for state_path, _ in entries}
        if len(model_by_key) != len(entries):
            raise ValueError(f"duplicate NNX state keys in {type(self).__name__}")

        loaded: dict[str, jnp.ndarray] = {}
        with safe_open(str(path), framework="numpy") as checkpoint:
            loaded = {key: jnp.asarray(checkpoint.get_tensor(key)) for key in checkpoint.keys()}

        checkpoint_keys = set(loaded)
        model_keys = set(model_by_key)
        missing_in_model = sorted(checkpoint_keys - model_keys)
        missing_in_checkpoint = sorted(model_keys - checkpoint_keys)
        if missing_in_model or missing_in_checkpoint:
            raise ValueError(
                f"checkpoint {path} state mismatch: unexpected={missing_in_model}, missing={missing_in_checkpoint}"
            )

        arrays: list[tuple[tuple[str | int, ...], jnp.ndarray]] = []
        for state_path, variable in entries:
            key = _state_key(state_path)
            array = loaded[key]
            if tuple(array.shape) != tuple(variable[...].shape):
                raise ValueError(
                    f"checkpoint {path} shape mismatch at {key}: model={variable[...].shape}, checkpoint={array.shape}"
                )
            arrays.append((state_path, array))
        nnx.update(self, nnx.from_flat_state(arrays))


__all__ = ["HubMixin"]
