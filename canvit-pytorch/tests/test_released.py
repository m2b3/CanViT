import pytest
import torch
from PIL import Image

from canvit_pytorch import CanViTForImageClassification, Viewpoint, sample_at_viewpoint
from canvit_pytorch.hub.repos import DINOV3_VITB16_IN1K_PROBE, FINETUNED_IN1K, FINETUNED_IN1K_NNX, FLAGSHIP
from canvit_pytorch.preprocess import preprocess

CAT_CLASSES = {281, 282, 285}  # ImageNet-1k: tabby cat, tiger cat, Egyptian cat


def _classify(model: CanViTForImageClassification, path: str) -> int:
    image = preprocess(512)(Image.open(path).convert("RGB"))
    assert isinstance(image, torch.Tensor)
    viewpoint = Viewpoint.full_scene(batch_size=1, device=None)
    glimpse = sample_at_viewpoint(spatial=image.unsqueeze(0), viewpoint=viewpoint, glimpse_size_px=128)
    with torch.inference_mode():
        logits, _ = model(glimpse=glimpse, state=model.init_state(batch_size=1, canvas_grid_size=32), viewpoint=viewpoint)
    return int(logits.argmax(dim=-1).item())


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("load", [
    lambda: CanViTForImageClassification.from_pretrained_with_probe(pretrained_repo=FLAGSHIP, probe_repo=DINOV3_VITB16_IN1K_PROBE),
    lambda: CanViTForImageClassification.from_pretrained(FINETUNED_IN1K),
    lambda: CanViTForImageClassification.from_pretrained(FINETUNED_IN1K_NNX["in21k"]),
    lambda: CanViTForImageClassification.from_pretrained(FINETUNED_IN1K_NNX["in1k"]),
], ids=["frozen-fused-probe", "fine-tuned", "fine-tuned-nnx-in21k", "fine-tuned-nnx-in1k"])
def test_released_classifiers_see_a_cat_in_one_glimpse(load):
    assert _classify(load().eval(), "test_data/Cat03.jpg") in CAT_CLASSES
