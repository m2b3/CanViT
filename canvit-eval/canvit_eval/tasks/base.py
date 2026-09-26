from dataclasses import dataclass
from pathlib import Path


@dataclass(kw_only=True)
class TaskConfig:
    output: Path
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 8
    amp: bool = True
