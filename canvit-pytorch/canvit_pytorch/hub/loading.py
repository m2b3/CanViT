"""Hugging Face Hub integration shared by CanViT's model classes: strict, typed loading."""

import dataclasses
import hashlib
import inspect
from pathlib import Path
from typing import Any, TypeVar

import safetensors.torch
from canvit_core.checkpoint import hub_file
from canvit_core.config import CanViTConfig
from huggingface_hub import HfApi, ModelHubMixin, PyTorchModelHubMixin
from torch import nn

from canvit_pytorch import legacy, project

T = TypeVar("T", bound=ModelHubMixin)

CODERS = {CanViTConfig: (dataclasses.asdict, lambda fields: CanViTConfig(**fields))}

FORMAT_VERSION = "0.2"
"""The canvit-pytorch release that introduced the checkpoint format written and read here."""

# The card save_pretrained writes into a local directory; published repos get theirs from hub.cards.
LOCAL_CARD_TEMPLATE = f"""---
{{{{ card_data }}}}
---

A checkpoint written by canvit-pytorch. {project.DESCRIPTION}

[Paper]({project.PAPER_URL}) · [Code]({project.CODE_URL})
"""


def checkpoint_record(repo: str, *, revision: str | None = None) -> dict[str, str | None]:
    """What identifies the checkpoint in `repo`: the Hub commit that `revision` (the latest when None) resolves to,
    None for a local directory, and the SHA-256 of its model.safetensors. Load it at the recorded revision."""
    if Path(repo).is_dir():
        assert revision is None, f"{repo} is a local directory, which has no revision"
        commit = None
    else:
        commit = HfApi().model_info(repo, revision=revision).sha
        assert commit, f"no Hub commit for {repo} at revision {revision}"
    with hub_file(repo, "model.safetensors", revision=commit).open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    return {"repo": repo, "revision": commit, "weights_sha256": digest}


class HubMixin(
    PyTorchModelHubMixin,
    library_name="canvit-pytorch",
    repo_url=project.CODE_URL,
    paper_url=project.PAPER_URL,
    license="mit",
    model_card_template=LOCAL_CARD_TEMPLATE,
):
    """Every weight of the model must come from the checkpoint, and every checkpoint weight must be used.

    `__init__` keyword arguments are the checkpoint's config.json; a
    CanViTConfig argument is stored as a nested object.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # The Hub mixin does not inherit coders.
        super().__init_subclass__(coders=CODERS, **kwargs)

    @classmethod
    def _from_pretrained(cls, *, model_id: str, revision: str | None = None, **kwargs: Any):
        required = [
            p.name for p in inspect.signature(cls.__init__).parameters.values()
            if p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty
        ]
        if missing := [name for name in required if name not in kwargs]:
            raise legacy.checkpoint_format_error(model_id, revision=revision, cls=cls, missing=missing)
        return super()._from_pretrained(model_id=model_id, revision=revision, **kwargs)

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        assert isinstance(model, nn.Module)
        missing, unexpected = safetensors.torch.load_model(model, model_file, strict=False, device=map_location)
        if missing or unexpected:
            raise ValueError(
                f"Weights in {model_file} do not match {type(model).__name__}: {len(missing)} missing, e.g. "
                f"{sorted(missing)[:3]}; {len(unexpected)} unexpected, e.g. {sorted(unexpected)[:3]}"
            )
        return model
