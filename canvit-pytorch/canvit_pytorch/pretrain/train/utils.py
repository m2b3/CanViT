"""Small local training helpers."""

import torch
from torch import Tensor, nn


def get_sensible_device() -> torch.device:
    """Pick the best available local torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(module: nn.Module) -> int:
    """Count trainable and frozen parameters."""
    return sum(p.numel() for p in module.parameters())


def assert_shape(tensor: Tensor, expected: tuple[int | None, ...]) -> None:
    """Assert tensor shape, with None acting as a wildcard dimension."""
    actual = tuple(tensor.shape)
    assert len(actual) == len(expected), f"Expected rank {len(expected)}, got shape {actual}"
    for idx, (got, want) in enumerate(zip(actual, expected, strict=True)):
        assert want is None or got == want, f"Shape mismatch at dim {idx}: expected {expected}, got {actual}"
