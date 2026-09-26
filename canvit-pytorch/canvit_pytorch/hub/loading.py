"""Hugging Face Hub integration shared by CanViT's model classes: strict, typed loading."""

import dataclasses
import inspect
from pathlib import Path
from typing import Any, TypeVar

import safetensors.torch
from huggingface_hub import ModelHubMixin, PyTorchModelHubMixin, hf_hub_download
from torch import nn

from canvit_pytorch import legacy, project
from canvit_pytorch.model.config import CanViTConfig

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


def hub_file(repo: str, filename: str) -> Path:
    """A file of a Hub repo, downloaded to the local cache, or of a local directory laid out like one."""
    return Path(repo) / filename if Path(repo).is_dir() else Path(hf_hub_download(repo, filename))


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
            raise legacy.weights_mismatch_error(model_file, missing=missing, unexpected=unexpected)
        return model
