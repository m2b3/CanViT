import torch
from safetensors.torch import save_file
from torch import nn

from canvit_pytorch import CanViTForImageClassification, CanViTForPretraining
from canvit_pytorch.model.classification import fuse_probe
from tests.model.tiny import TINY

TEACHER_DIM = 24


def test_fuse_probe_equals_projection_destandardization_and_probe():
    torch.manual_seed(0)
    proj, probe = nn.Linear(16, TEACHER_DIM), nn.Linear(TEACHER_DIM, 5)
    mean, std = torch.randn(TEACHER_DIM), torch.rand(TEACHER_DIM) + 0.1
    weight, bias = fuse_probe(proj=proj, mean=mean, std=std, probe_weight=probe.weight, probe_bias=probe.bias)
    z = torch.randn(4, 16)
    with torch.no_grad():
        torch.testing.assert_close(z @ weight.T + bias, probe(std * proj(z) + mean))


def test_logits_stay_float32_under_bfloat16_autocast(glimpses):
    # Linear and LayerNorm re-cast their inputs under autocast; a bfloat16 head flips a fraction of predictions.
    glimpse, viewpoint = glimpses
    model = CanViTForImageClassification(canvit_config=TINY, n_classes=10).eval()
    with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits, _ = model(glimpse=glimpse, state=model.init_state(batch_size=2, canvas_grid_size=4), viewpoint=viewpoint)
    assert logits.dtype == torch.float32 and logits.shape == (2, 10)


def test_from_pretrained_with_probe_matches_the_unfused_readout(tmp_path, glimpses):
    torch.manual_seed(0)
    pretrained = CanViTForPretraining(canvit_config=TINY, teacher_dim=TEACHER_DIM, teacher_patch_grid=4).eval()
    pretrained.teacher_cls_standardizer.fit(3 * torch.randn(64, 1, TEACHER_DIM) + 1)
    pretrained.teacher_patch_standardizer.fit(torch.randn(64, 16, TEACHER_DIM))
    pretrained.save_pretrained(tmp_path / "pretrained")
    probe = nn.Linear(TEACHER_DIM, 7)
    (tmp_path / "probe").mkdir()
    save_file({"weight": probe.weight.detach().clone(), "bias": probe.bias.detach().clone()}, tmp_path / "probe" / "model.safetensors")

    model = CanViTForImageClassification.from_pretrained_with_probe(
        pretrained_repo=str(tmp_path / "pretrained"), probe_repo=str(tmp_path / "probe")).eval()
    glimpse, viewpoint = glimpses
    with torch.inference_mode():
        logits, _ = model(glimpse=glimpse, state=model.init_state(batch_size=2, canvas_grid_size=4), viewpoint=viewpoint)
        cls = pretrained(glimpse=glimpse, state=pretrained.init_state(batch_size=2, canvas_grid_size=4), viewpoint=viewpoint)
        teacher_cls = pretrained.teacher_cls_standardizer.destandardize(pretrained.predict_teacher_cls(cls.state.recurrent_cls))
        torch.testing.assert_close(logits, probe(teacher_cls), atol=1e-4, rtol=1e-4)
