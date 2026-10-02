import json
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from canvit_core import CanViTConfig
from safetensors.numpy import save_file

from tests.backends.support import Backend
from tools.checkpoint_conversion import native_state_arrays


@pytest.mark.parametrize("kind", ["classification", "pretraining"])
def test_checkpoint_roundtrip_preserves_native_state(
    backend: Backend, tiny_config: CanViTConfig, tmp_path: Path, kind: Literal["classification", "pretraining"],
):
    if kind == "classification":
        model = backend.make_classifier(tiny_config, n_classes=7)
    else:
        model = backend.make_pretraining(tiny_config, teacher_dim=12, teacher_patch_grid=4)
    path = tmp_path / kind
    model.save_pretrained(path)
    loaded = type(model).from_pretrained(path)
    expected = native_state_arrays(model, backend=backend.name)
    actual = native_state_arrays(loaded, backend=backend.name)
    assert set(expected) == set(actual)
    for name, value in expected.items():
        np.testing.assert_array_equal(value, actual[name], err_msg=name)


def test_checkpoint_rejects_missing_or_unexpected_weight(
    backend: Backend, tiny_config: CanViTConfig, tmp_path: Path,
):
    model = backend.make_classifier(tiny_config, n_classes=7)
    source = tmp_path / "source"
    model.save_pretrained(source)
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    shutil.copyfile(source / "config.json", malformed / "config.json")
    arrays = native_state_arrays(model, backend=backend.name)
    arrays.pop(next(iter(arrays)))
    arrays["unexpected"] = np.zeros((1,), dtype=np.float32)
    save_file(arrays, malformed / "model.safetensors")
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        type(model).from_pretrained(malformed)


def test_checkpoint_rejects_malformed_config(backend: Backend, tiny_config: CanViTConfig, tmp_path: Path):
    model = backend.make_classifier(tiny_config, n_classes=7)
    source = tmp_path / "source"
    model.save_pretrained(source)
    record = json.loads((source / "config.json").read_text())
    record["format"] = "malformed"
    (source / "config.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        type(model).from_pretrained(source)


def test_checkpoint_rejects_string_boolean_in_canvit_config(backend: Backend, tiny_config: CanViTConfig, tmp_path: Path):
    model = backend.make_classifier(tiny_config, n_classes=7)
    source = tmp_path / "source"
    model.save_pretrained(source)
    record = json.loads((source / "config.json").read_text())
    record["config"]["canvit_config"]["enable_vpe"] = "false"
    (source / "config.json").write_text(json.dumps(record))
    with pytest.raises((AssertionError, TypeError, ValueError)):
        type(model).from_pretrained(source)
