"""The reference: AdaGlimpse's own torch modules at the pinned commit, and their weights as this package's."""

import importlib
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from torch import nn

from adaglimpse_nnx import upstream

FIXED_TABLES = frozenset({"pos_embed", "decoder_pos_embed"})
"""Upstream's sine-cosine tables, stored as frozen parameters; this package computes the decoder's and never uses
the encoder's, which upstream reads only for patches given without boxes."""


def _git(*args: str | Path) -> str:
    return subprocess.run(("git", *map(str, args)), check=True, capture_output=True, text=True).stdout.strip()


def checkout(cache_dir: Path) -> Path:
    """A clean checkout of AdaGlimpse at the pinned commit under cache_dir, fetched on first use."""
    path = cache_dir / f"AdaGlimpse-{upstream.COMMIT}"
    if not path.exists():
        partial = path.with_name(path.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        _git("init", "--quiet", partial)
        _git("-C", partial, "fetch", "--quiet", "--depth", "1", upstream.REPOSITORY, upstream.COMMIT)
        _git("-C", partial, "checkout", "--quiet", "--detach", "FETCH_HEAD")
        partial.rename(path)
    head, changes = _git("-C", path, "rev-parse", "HEAD"), _git("-C", path, "status", "--porcelain")
    assert head == upstream.COMMIT and not changes, f"{path}: HEAD {head}, changes {changes!r}"
    return path


def import_modules(source: Path) -> types.SimpleNamespace:
    """AdaGlimpse's model module (`architectures.mae`), its positional embeddings (`architectures.mae_utils`) and
    its glimpse sampler (`architectures.rl.interactive_sampler`).

    The packages' `__init__` files import the whole training stack (Lightning,
    TorchRL), so the packages are registered bare; the sampler's one import
    from TorchRL's world, `SharedMemory`, only annotates a method these tests do
    not call.
    """
    for name in ("architectures", "architectures.rl", "datasets"):
        package = types.ModuleType(name)
        package.__path__ = [str(source / name.replace(".", "/"))]
        sys.modules[name] = package
    shared_memory = types.ModuleType("architectures.rl.shared_memory")
    vars(shared_memory)["SharedMemory"] = object
    sys.modules[shared_memory.__name__] = shared_memory
    return types.SimpleNamespace(
        mae=importlib.import_module("architectures.mae"),
        mae_utils=importlib.import_module("architectures.mae_utils"),
        sampler=importlib.import_module("architectures.rl.interactive_sampler"),
    )


def package_tensors(model: nn.Module) -> dict[str, np.ndarray]:
    """An upstream MaskedAutoencoderViT's parameters, keyed and laid out as this package's state files hold them."""
    tensors: dict[str, np.ndarray] = {}
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            if name in FIXED_TABLES:
                continue
            value = parameter.detach().numpy()
            match module, name:
                # The patch embedding: a convolution upstream (`patch_embed.proj`), a linear map of NHWC pixels here.
                case nn.Conv2d(), "weight":  # [out, channels, height, width] -> [(height, width, channels), out]
                    assert module_name == "patch_embed.proj", module_name
                    key, value = "patch_embed.kernel", value.transpose(2, 3, 1, 0).reshape(-1, value.shape[0])
                case nn.Conv2d(), "bias":
                    key = "patch_embed.bias"
                case nn.Linear(), "weight":
                    key, value = f"{module_name}.kernel", value.T
                case nn.LayerNorm(), "weight":
                    key = f"{module_name}.scale"
                case _:
                    key = f"{module_name}.{name}" if module_name else name
            assert key not in tensors, key
            tensors[key] = np.ascontiguousarray(value)
    return tensors
