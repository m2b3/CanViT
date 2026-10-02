import json
from pathlib import Path

import numpy as np
import pytest
import torch
from canvit_pytorch import SegmentationProbe as TorchSegmentationProbe

from tests.backends.support import Backend, relative_l2
from tools.checkpoint_conversion import convert_named_arrays, load_native_arrays, native_state_arrays
from tools.convert_checkpoints import ConvertCheckpoints, run


def _load_probe_weights(backend: Backend, native, reference: TorchSegmentationProbe) -> None:
    source = {name: value.detach().cpu().numpy() for name, value in reference.state_dict().items()}
    schema = {name: tuple(value.shape) for name, value in native_state_arrays(native, backend=backend.name).items()}
    converted = convert_named_arrays(source, target_shapes=schema, backend=backend.name)
    load_native_arrays(native, converted, backend=backend.name)


def _set_reference_batch_stats(probe: TorchSegmentationProbe) -> None:
    assert probe.bn.running_mean is not None and probe.bn.running_var is not None
    with torch.no_grad():
        probe.bn.running_mean.copy_(torch.linspace(-0.4, 0.4, probe.embed_dim))
        probe.bn.running_var.copy_(torch.linspace(0.7, 1.4, probe.embed_dim))
        probe.bn.weight.copy_(torch.linspace(0.8, 1.2, probe.embed_dim))
        probe.bn.bias.copy_(torch.linspace(-0.2, 0.2, probe.embed_dim))


@pytest.mark.parametrize("use_ln", [False, True])
def test_segmentation_probe_eval_matches_pytorch(backend: Backend, use_ln: bool):
    torch.manual_seed(3021)
    reference = TorchSegmentationProbe(embed_dim=8, num_classes=3, dropout=0.0, use_ln=use_ln).eval()
    _set_reference_batch_stats(reference)
    native = backend.make_probe(embed_dim=8, num_classes=3, dropout=0.0, use_ln=use_ln)
    _load_probe_weights(backend, native, reference)
    backend.set_probe_eval(native)
    features = np.random.default_rng(3021).normal(size=(2, 5, 7, 8)).astype(np.float32)
    with torch.inference_mode():
        expected = reference(torch.from_numpy(features)).permute(0, 2, 3, 1).numpy()
    actual = backend.to_numpy(native(backend.array(features)))
    assert relative_l2(expected, actual) < 2e-5
    np.testing.assert_allclose(expected, actual, atol=3e-6, rtol=3e-5)


def test_segmentation_probe_training_updates_batch_stats_like_pytorch(backend: Backend):
    torch.manual_seed(3022)
    reference = TorchSegmentationProbe(embed_dim=8, num_classes=3, dropout=0.0, use_ln=True).train()
    _set_reference_batch_stats(reference)
    native = backend.make_probe(embed_dim=8, num_classes=3, dropout=0.0, use_ln=True)
    _load_probe_weights(backend, native, reference)
    backend.set_probe_train(native)
    features = np.random.default_rng(3022).normal(size=(2, 5, 7, 8)).astype(np.float32)
    with torch.no_grad():
        expected = reference(torch.from_numpy(features)).permute(0, 2, 3, 1).numpy()
    actual = backend.to_numpy(native(backend.array(features)))
    assert relative_l2(expected, actual) < 2e-5
    np.testing.assert_allclose(expected, actual, atol=3e-6, rtol=3e-5)
    native_state = native_state_arrays(native, backend=backend.name)
    assert reference.bn.running_mean is not None and reference.bn.running_var is not None
    assert reference.bn.num_batches_tracked is not None
    np.testing.assert_allclose(native_state["bn.running_mean"], reference.bn.running_mean.numpy(), atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(native_state["bn.running_var"], reference.bn.running_var.numpy(), atol=3e-6, rtol=3e-5)
    assert int(np.asarray(native_state["bn.num_batches_tracked"])) == int(reference.bn.num_batches_tracked)


def test_probe_converter_roundtrip(backend: Backend, tmp_path: Path):
    torch.manual_seed(3023)
    source_model = TorchSegmentationProbe(embed_dim=8, num_classes=3, dropout=0.0, use_ln=True).eval()
    source = tmp_path / "source"
    source_model.save_pretrained(source)
    output = tmp_path / backend.name
    run(ConvertCheckpoints(source=str(source), output=output, backend=backend.name, kind="probe"))
    record = json.loads((output / "conversion.json").read_text())
    assert record["kind"] == "probe"
    assert record["metrics_after_reload"]["probe.logits"]["relative_l2"] < 5e-5
