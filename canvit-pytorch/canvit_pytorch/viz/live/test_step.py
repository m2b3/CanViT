import pytest
import torch

pytest.importorskip("onnxscript")
ort = pytest.importorskip("onnxruntime")

from canvit_pytorch import CanViT, CanViTConfig, CanViTForSemanticSegmentation, SegmentationProbe  # noqa: E402
from canvit_pytorch.policies.entropy import predictive_entropy  # noqa: E402
from canvit_pytorch.viz.live.export import export_graph  # noqa: E402
from canvit_pytorch.viz.live.step import (  # noqa: E402
    GlimpseStep,
    StepInputs,
    StepOutputs,
    predictive_entropy_via_logsumexp,
)


def test_entropy_via_logsumexp_is_predictive_entropy():
    logits = torch.randn(2, 150, 4, 4) * 30  # from nearly uniform to nearly one-hot cells
    torch.testing.assert_close(predictive_entropy_via_logsumexp(logits), predictive_entropy(logits), rtol=1e-5, atol=1e-5)


def test_onnx_graph_computes_the_pytorch_step(tmp_path):
    torch.manual_seed(0)
    canvit = CanViT(CanViTConfig(backbone_name="vits16", canvas_num_heads=2, canvas_head_dim=32, num_canvas_registers=2))
    probe = SegmentationProbe(embed_dim=canvit.canvas_dim, num_classes=7, dropout=0.0, use_ln=True)
    model = CanViTForSemanticSegmentation(canvit=canvit, probe=probe).eval()
    with torch.no_grad():  # random weights everywhere, so every path (LayerScale starts at 1e-5) reaches the outputs
        for parameter in model.parameters():
            parameter.normal_(std=0.2)
    grid = 8
    # An off-center viewpoint at a scale no quadtree uses, on a canvas other than the initial one.
    inputs = StepInputs(
        scene=torch.randn(1, 3, 64, 64),
        canvas=torch.randn(1, canvit.config.num_canvas_registers + grid * grid, canvit.canvas_dim),
        recurrent_cls=torch.randn(1, 1, canvit.backbone_dim),
        centers=torch.tensor([[0.31, -0.42]]),
        scales=torch.tensor([0.37]),
    )
    path = tmp_path / "step.onnx"
    expected = export_graph(GlimpseStep(model, glimpse_size_px=32).eval(), inputs, path)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {name: value.numpy() for name, value in inputs._asdict().items()}
    actual = StepOutputs(*session.run(list(StepOutputs._fields), feeds))
    for name in StepOutputs._fields:
        torch.testing.assert_close(
            torch.from_numpy(getattr(actual, name)), getattr(expected, name), rtol=1e-4, atol=1e-4, msg=name,
        )
