import pytest
import torch

pytest.importorskip("onnxscript")
ort = pytest.importorskip("onnxruntime")

from canvit_pytorch import CanViT, CanViTConfig, CanViTForSemanticSegmentation, SegmentationProbe  # noqa: E402
from canvit_pytorch.policies.entropy import predictive_entropy  # noqa: E402
from canvit_pytorch.viz.live.export import export_graph  # noqa: E402
from canvit_pytorch.viz.live.step import (  # noqa: E402
    GlimpseStep,
    ProbeReadout,
    ReadoutInputs,
    ReadoutOutputs,
    StepInputs,
    StepOutputs,
    predictive_entropy_via_logsumexp,
)

GRID = 8


def test_entropy_via_logsumexp_is_predictive_entropy():
    logits = torch.randn(2, 150, 4, 4) * 30  # from nearly uniform to nearly one-hot cells
    torch.testing.assert_close(predictive_entropy_via_logsumexp(logits), predictive_entropy(logits), rtol=1e-5, atol=1e-5)


@pytest.fixture
def model() -> CanViTForSemanticSegmentation:
    torch.manual_seed(0)
    canvit = CanViT(CanViTConfig(backbone_name="vits16", canvas_num_heads=2, canvas_head_dim=32, num_canvas_registers=2))
    probe = SegmentationProbe(embed_dim=canvit.canvas_dim, num_classes=7, dropout=0.0, use_ln=True)
    model = CanViTForSemanticSegmentation(canvit=canvit, probe=probe).eval()
    with torch.no_grad():  # random weights everywhere, so every path (LayerScale starts at 1e-5) reaches the outputs
        for parameter in model.parameters():
            parameter.normal_(std=0.2)
    return model


def canvas(model: CanViTForSemanticSegmentation) -> torch.Tensor:
    canvit = model.canvit
    return torch.randn(1, canvit.config.num_canvas_registers + GRID * GRID, canvit.canvas_dim)


def run_onnx(path, names, inputs) -> list:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(list(names), {name: value.numpy() for name, value in zip(inputs._fields, inputs, strict=True)})


def test_step_graph_computes_the_pytorch_step(model, tmp_path):
    # An off-center viewpoint at a scale no quadtree uses, on a canvas other than the initial one.
    inputs = StepInputs(
        scene=torch.randn(1, 3, 64, 64), canvas=canvas(model), recurrent_cls=torch.randn(1, 1, model.canvit.backbone_dim),
        centers=torch.tensor([[0.31, -0.42]]), scales=torch.tensor([0.37]),
    )
    path = tmp_path / "step.onnx"
    expected = export_graph(GlimpseStep(model.canvit, glimpse_size_px=32).eval(), inputs, StepInputs._fields,
                            StepOutputs._fields, path)
    for name, actual, wanted in zip(StepOutputs._fields, run_onnx(path, StepOutputs._fields, inputs), expected, strict=True):
        torch.testing.assert_close(torch.from_numpy(actual), wanted, rtol=1e-4, atol=1e-4, msg=name)


def test_probe_graph_computes_the_pytorch_readout_with_the_probe_alone(model, tmp_path):
    inputs = ReadoutInputs(canvas(model))
    path = tmp_path / "probe.onnx"
    expected = export_graph(ProbeReadout(model).eval(), inputs, ReadoutInputs._fields, ReadoutOutputs._fields, path)
    for name, actual, wanted in zip(ReadoutOutputs._fields, run_onnx(path, ReadoutOutputs._fields, inputs), expected,
                                    strict=True):
        torch.testing.assert_close(torch.from_numpy(actual), wanted, rtol=1e-4, atol=1e-4, msg=name)
    # CanViT's weights stay out of the probe's graph: it is little bigger than the probe's parameters.
    probe_bytes = 4 * sum(p.numel() for p in model.probe.parameters()) + 4 * sum(b.numel() for b in model.probe.buffers())
    assert path.stat().st_size < probe_bytes + 64_000, (path.stat().st_size, probe_bytes)
