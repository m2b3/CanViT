import pytest
import torch

from canvit_pytorch import CanViTForPretraining, CanViTForSemanticSegmentation, SegmentationProbe
from tests.model.tiny import TINY

RELEASED_PROBE_KEYS = {
    "ln.weight", "ln.bias", "bn.weight", "bn.bias", "bn.running_mean", "bn.running_var", "bn.num_batches_tracked",
    "conv.weight", "conv.bias",
}


def test_probe_weights_have_the_names_of_the_released_probes():
    assert set(SegmentationProbe(embed_dim=32, num_classes=10, dropout=0.0, use_ln=True).state_dict()) == RELEASED_PROBE_KEYS
    without_ln = SegmentationProbe(embed_dim=32, num_classes=10, dropout=0.0, use_ln=False)
    assert set(without_ln.state_dict()) == RELEASED_PROBE_KEYS - {"ln.weight", "ln.bias"}


def test_probe_rejects_features_of_another_width():
    probe = SegmentationProbe(embed_dim=32, num_classes=10, dropout=0.0, use_ln=True)
    with pytest.raises(AssertionError, match="probe expects 32-dim features"):
        probe(torch.randn(1, 4, 4, 16))


def test_from_pretrained_with_probe_decodes_the_canvas(tmp_path, glimpses):
    torch.manual_seed(0)
    pretrained = CanViTForPretraining(canvit_config=TINY, teacher_dim=24, teacher_patch_grid=4).eval()
    pretrained.save_pretrained(tmp_path / "pretrained")
    probe = SegmentationProbe(embed_dim=TINY.canvas_dim, num_classes=7, dropout=0.0, use_ln=True).eval()
    assert probe.bn.running_mean is not None and probe.bn.running_var is not None
    probe.bn.running_mean.normal_()
    probe.bn.running_var.uniform_(0.5, 2)
    probe.save_pretrained(tmp_path / "probe")

    model = CanViTForSemanticSegmentation.from_pretrained_with_probe(
        pretrained_repo=str(tmp_path / "pretrained"), probe_repo=str(tmp_path / "probe")).eval()
    glimpse, viewpoint = glimpses
    with torch.inference_mode():
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits, state = model(glimpse=glimpse, state=model.init_state(batch_size=2, canvas_grid_size=4), viewpoint=viewpoint)
        assert logits.dtype == torch.float32 and logits.shape == (2, 7, 4, 4)
        logits, state = model(glimpse=glimpse, state=model.init_state(batch_size=2, canvas_grid_size=4), viewpoint=viewpoint)
        torch.testing.assert_close(logits, probe(pretrained.canvit.canvas_patch_grid(state.canvas)))
        upsampled, _ = model.predict(glimpse=glimpse, state=model.init_state(batch_size=2, canvas_grid_size=4),
                                     viewpoint=viewpoint, target_size=(24, 40))
    assert upsampled.shape == (2, 7, 24, 40)
