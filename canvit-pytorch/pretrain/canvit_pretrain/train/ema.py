"""Exponential moving average tracker for training metrics."""

from collections.abc import ItemsView

from torch import Tensor


class EMATracker:
    """Track exponential moving averages of arbitrary metrics."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._values: dict[str, Tensor] = {}

    def update(self, key: str, val: Tensor) -> Tensor:
        """Update EMA for key with new value. Returns updated EMA."""
        # Store in float32 to avoid bf16 numerical issues
        v = val.detach().float()
        if key not in self._values:
            self._values[key] = v
        else:
            self._values[key] = self.alpha * v + (1 - self.alpha) * self._values[key]
        return self._values[key]

    def get(self, key: str) -> Tensor | None:
        """Get current EMA value for key, or None if not tracked."""
        return self._values.get(key)

    def items(self) -> ItemsView[str, Tensor]:
        """Get all current EMA key-value pairs."""
        return self._values.items()
