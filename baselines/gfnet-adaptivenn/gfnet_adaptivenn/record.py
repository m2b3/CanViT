"""What every measurement record states about how it was made."""

from datetime import datetime
from typing import Any

import torch

from gfnet_adaptivenn.sources import Repository


def provenance(repository: Repository, *, device: torch.device) -> dict[str, Any]:
    """The code measured, the PyTorch version and device that measured it, and when."""
    return {
        "code": {"repository": repository.url, "commit": repository.commit, "patch": repository.patch},
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
        "measured": datetime.now().astimezone().isoformat(timespec="minutes"),
    }
