"""Fixed inputs, and a checkpoint's outputs on them, that prove a format conversion exact.

A checkpoint converted to the 0.2 format must reproduce, bit for bit, what the 0.1 code computed for it on these
inputs (hub.republish for the repos released in 0.1, hub.publish for classifiers exported in the 0.1 format;
scripts/record_0_1_outputs.py records the 0.1 outputs).
"""

import hashlib
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image
from torch import Tensor

from canvit_pytorch.hub.repos import RELEASED_CANVAS_GRID_SIZE, RELEASED_GLIMPSE_SIZE_PX, RELEASED_SCENE_SIZE_PX
from canvit_pytorch.model.classification import CanViTForImageClassification
from canvit_pytorch.model.pretraining import CanViTForPretraining
from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint

Kind = Literal["pretraining", "classification", "probe"]

IMAGE = Path(__file__).parents[2] / "test_data" / "Cat03.jpg"
VIEWPOINTS = [(0.0, 0.0, 1.0), (-0.5, 0.5, 0.5), (0.3, -0.2, 0.25)]  # (row, col, scale)


def inputs() -> dict[str, Any]:
    """The inputs as a reference file records them."""
    return {"image_sha256": hashlib.sha256(IMAGE.read_bytes()).hexdigest(), "viewpoints": VIEWPOINTS,
            "scene_size_px": RELEASED_SCENE_SIZE_PX, "glimpse_size_px": RELEASED_GLIMPSE_SIZE_PX,
            "canvas_grid_size": RELEASED_CANVAS_GRID_SIZE}


def _rollout(model: CanViTForPretraining | CanViTForImageClassification, image: Tensor) -> dict[str, Tensor]:
    state = model.init_state(batch_size=1, canvas_grid_size=RELEASED_CANVAS_GRID_SIZE)
    outputs: dict[str, Tensor] = {}
    for row, col, scale in VIEWPOINTS:
        viewpoint = Viewpoint(centers=torch.tensor([[row, col]]), scales=torch.tensor([scale]))
        glimpse = sample_at_viewpoint(spatial=image, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
        if isinstance(model, CanViTForImageClassification):
            outputs["logits"], state = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
        else:
            state = model(glimpse=glimpse, state=state, viewpoint=viewpoint).state
    outputs["canvas"] = state.canvas
    if isinstance(model, CanViTForPretraining):
        outputs |= {"cls": state.recurrent_cls, "patches": model.predict_teacher_patches(state.canvas),
                    "cls_pred": model.predict_teacher_cls(state.recurrent_cls)}
    return outputs


@torch.inference_mode()
def verify(staged: Path, kind: Kind, reference: dict[str, Tensor]) -> None:
    """The staged checkpoint, loaded strictly by 0.2 code, reproduces the 0.1 outputs bit for bit."""
    assert reference["kind"] == kind, (reference["kind"], kind)
    if kind == "probe":
        probe = SegmentationProbe.from_pretrained(str(staged)).eval()
        outputs = {"logits": probe(reference["features"])}
    else:
        model_class = CanViTForPretraining if kind == "pretraining" else CanViTForImageClassification
        model = model_class.from_pretrained(str(staged)).eval()
        image = preprocess(RELEASED_SCENE_SIZE_PX)(Image.open(IMAGE).convert("RGB"))
        assert isinstance(image, Tensor)
        outputs = _rollout(model, image.unsqueeze(0))
    for name, value in outputs.items():
        assert torch.equal(value, reference[name]), f"{staged.name}: {name} differs from the 0.1 output"
