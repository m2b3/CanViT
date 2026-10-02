import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Self

from huggingface_hub import hf_hub_download, snapshot_download

FORMAT = "d63be48b-fd67-4299-b1c7-c7b3a2b3a238"


def hub_file(repo: str | Path, filename: str, *, revision: str | None = None) -> Path:
    return (
        Path(repo) / filename if Path(repo).is_dir() else Path(hf_hub_download(str(repo), filename, revision=revision))
    )


class CheckpointMixin(ABC):
    backend: ClassVar[str]

    @abstractmethod
    def checkpoint_config(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> Self: ...

    @abstractmethod
    def _save_weights(self, path: Path) -> None: ...

    @abstractmethod
    def _load_weights(self, path: Path) -> None: ...

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._save_weights(directory / "model.safetensors")
        record = dict(format=FORMAT, backend=self.backend, model=type(self).__name__, config=self.checkpoint_config())
        (directory / "config.json").write_text(json.dumps(record, indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, repo: str | Path, *, revision: str | None = None) -> Self:
        directory = Path(repo)
        if not directory.is_dir():
            snapshot = snapshot_download(str(repo), revision=revision, allow_patterns=["config.json", "model.safetensors"])
            assert isinstance(snapshot, str)
            directory = Path(snapshot)
        record = json.loads((directory / "config.json").read_text())
        if record.get("format") != FORMAT:
            raise ValueError(f"{repo}: unsupported checkpoint format; re-export with the CanViT checkpoint converter")
        if record["backend"] != cls.backend or record["model"] != cls.__name__:
            raise ValueError(
                f"{repo}: expected {cls.backend}/{cls.__name__}, got {record['backend']}/{record['model']}"
            )
        model = cls.from_config(record["config"])
        model._load_weights(directory / "model.safetensors")
        return model
