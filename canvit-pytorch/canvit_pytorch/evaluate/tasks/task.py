import dataclasses
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from canvit_pytorch.progress import progress
from canvit_pytorch.provenance import provenance

log = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Task:
    """Settings shared by every evaluation task."""

    output: Path
    batch_size: int
    device: str = "cuda"
    num_workers: int = 8
    """DataLoader worker processes."""
    amp: bool = True
    """Run CanViT and DINOv3 under bfloat16 autocast; probes and metrics compute in float32."""

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def autocast(self) -> torch.autocast:
        return torch.autocast(device_type=self.torch_device.type, dtype=torch.bfloat16, enabled=self.amp)

    def batches(self, dataset: Dataset, *, description: str) -> Iterable[Any]:
        """The dataset in order, batched, reporting progress."""
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            pin_memory=self.torch_device.type == "cuda",
        )
        return progress(loader, description=description)

    def metadata(self, **entries: object) -> dict[str, object]:
        """What produced a result: this configuration, the code and runtime that ran it, and entries."""
        entries = {"config": dataclasses.asdict(self), "provenance": provenance(self.torch_device), **entries}
        return {key: _plain(value) for key, value in entries.items()}

    def save(self, result: dict[str, object], **metadata_entries: object) -> Path:
        """torch.save result, with a "metadata" entry, to self.output."""
        self.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**result, "metadata": self.metadata(**metadata_entries)}, self.output)
        log.info("Saved %s", self.output)
        return self.output


def _plain(value: object) -> object:
    """value with builtins only: every Path and str subclass (such as torch.__version__) as a str, every list or
    tuple subclass as a list or tuple. A pickled class is named by its module, which can move (pathlib's did in
    Python 3.13); with builtins and tensors only, a result loads with torch.load(weights_only=True) on any Python."""
    match value:
        case Path() | str():
            return str(value)
        case dict():
            return {key: _plain(item) for key, item in value.items()}
        case list():
            return [_plain(item) for item in value]
        case tuple():
            return tuple(_plain(item) for item in value)
        case _:
            return value
