"""The released GFNet and AdaptiveNN code and checkpoints the measurements run, pinned by commit and SHA-256.

    uv run python -m gfnet_adaptivenn.sources --root <directory>

clones each repository at its commit and downloads each checkpoint from the
Google Drive link of the repository's README, verifying every file; a
complete root is verified and left unchanged.
"""

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import gdown
import tyro


@dataclass(frozen=True)
class Repository:
    url: str
    commit: str
    directory: str
    # GFNet imports torch._six, which torch 2.0 removed; it re-exported collections.abc, so the import binds the same.
    patch: tuple[str, str, str] | None = None  # (file, line, replacement)


@dataclass(frozen=True)
class Checkpoint:
    file: str
    drive_id: str
    sha256: str

    @property
    def url(self) -> str:
        return f"https://drive.google.com/file/d/{self.drive_id}"


GFNET = Repository(
    url="https://github.com/blackfeather-wang/GFNet-Pytorch",
    commit="8a6775c8267c644aceb104796aebe2baab0dd595",
    directory="GFNet-Pytorch",
    patch=("models/conv2d_layers.py", "from torch._six import container_abcs",
           "import collections.abc as container_abcs"),
)
ADAPTIVENN = Repository(
    url="https://github.com/LeapLabTHU/AdaptiveNN",
    commit="020bf8d33271d7b02e2e63829e4b13d987868b89",
    directory="AdaptiveNN",
)
GFNET_CHECKPOINTS = (
    Checkpoint("gf_effb3_144_T4.pth.tar", "1OVTGI6d2nsN5Hz5T_qLnYeUIBL5oMeeU",
               "993af87470ce8c594324ccc7c93e4f7687f672d9d0d29012e32448b016333003"),
    Checkpoint("gf_mbv3l100_128_T3.pth.tar", "1Ud_olyer-YgAb667YUKs38C2O1Yb6Nom",
               "b52bf5cabb9665e3b5c9f879a699c9533ab9eea4ad0ccaeed50688d101686470"),
)
"""The GFNet models of the paper's ImageNet-1k comparison: EfficientNet-B3 and MobileNet-V3-Large backbones."""
ADAPTIVENN_CHECKPOINT = Checkpoint(
    "adaptivenn_ckpt.pth", "1OIPZyTozJatdd40VZKaJHsw0ShaujQ0Y",
    "b7413f83fb5b99aeba76ff60c66c3049c6c4f4232c6f530e5e414ee214b2a316",
)
"""AdaptiveNN-DeiT-S trained on ImageNet-1k."""


@dataclass(frozen=True)
class Sources:
    """A root holding each repository's checkout and a `checkpoints` directory, every file verified."""

    root: Path

    def repository(self, repository: Repository) -> Path:
        path = self.root / repository.directory
        _verify_checkout(repository, path)
        return path

    def checkpoint(self, checkpoint: Checkpoint) -> Path:
        path = self.root / "checkpoints" / checkpoint.file
        digest = _sha256(path)
        if digest != checkpoint.sha256:
            raise ValueError(f"{path}: SHA-256 {digest}, pinned {checkpoint.sha256}")
        return path


def fetch(root: Path) -> Sources:
    root.mkdir(parents=True, exist_ok=True)
    for repository in (GFNET, ADAPTIVENN):
        path = root / repository.directory
        if not path.exists():
            _git("clone", "--quiet", repository.url, path)
            _git("-C", path, "checkout", "--quiet", "--detach", repository.commit)
            if repository.patch is not None:
                file, line, replacement = repository.patch
                source = path / file
                text = source.read_text()
                assert text.count(line) == 1, f"{source}: expected the line {line!r} once"
                source.write_text(text.replace(line, replacement))
    for checkpoint in (*GFNET_CHECKPOINTS, ADAPTIVENN_CHECKPOINT):
        path = root / "checkpoints" / checkpoint.file
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".partial")
            gdown.download(id=checkpoint.drive_id, output=str(partial), quiet=False)
            partial.rename(path)
    sources = Sources(root)
    for repository in (GFNET, ADAPTIVENN):
        sources.repository(repository)
    for checkpoint in (*GFNET_CHECKPOINTS, ADAPTIVENN_CHECKPOINT):
        sources.checkpoint(checkpoint)
    return sources


def _verify_checkout(repository: Repository, path: Path) -> None:
    """HEAD is the pinned commit, and the working tree differs from it by the pinned patch alone."""
    head = _git("-C", path, "rev-parse", "HEAD")
    if head != repository.commit:
        raise ValueError(f"{path}: HEAD {head}, pinned {repository.commit}")
    changes = [line for line in _git("-C", path, "diff", "-U0", "HEAD").splitlines()
               if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    expected = [] if repository.patch is None else [f"-{repository.patch[1]}", f"+{repository.patch[2]}"]
    if changes != expected:
        raise ValueError(f"{path}: tracked files differ from {repository.commit} by {changes}, expected {expected}")


def _git(*args: str | Path) -> str:
    return subprocess.run(("git", *map(str, args)), check=True, capture_output=True, text=True).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Fetch:
    root: Path
    """Directory for the checkouts and checkpoints."""


if __name__ == "__main__":
    fetch(tyro.cli(Fetch).root)
