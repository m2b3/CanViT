import pytest
import torch

from canvit_pytorch import CanViT, CanViTOutput, RecurrentState, Viewpoint
from canvit_pytorch.model.canvit import canvas_attention_schedule
from tests.model.tiny import TINY, glimpse_batch


@pytest.fixture(scope="module")
def model() -> CanViT:
    torch.manual_seed(0)
    return CanViT(TINY).eval()


def test_canvit_b_reads_and_writes_three_times_per_glimpse():
    assert canvas_attention_schedule(num_blocks=12, rw_stride=2, enable_reads=True) == ((1, 5, 9), (3, 7, 11))
    assert canvas_attention_schedule(num_blocks=12, rw_stride=2, enable_reads=False) == ((), (3, 7, 11))


@pytest.mark.parametrize("canvas_grid_size", [4, 8])
def test_the_canvas_grid_is_chosen_at_inference(model, glimpses, canvas_grid_size):
    glimpse, viewpoint = glimpses
    state = model.init_state(batch_size=2, canvas_grid_size=canvas_grid_size)
    assert state.canvas.shape == (2, TINY.num_canvas_registers + canvas_grid_size**2, TINY.canvas_dim)
    with torch.inference_mode():
        out = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
    assert out.state.canvas.shape == state.canvas.shape
    assert out.state.recurrent_cls.shape == (2, 1, model.backbone_dim)
    assert not torch.equal(out.state.canvas, state.canvas)


def test_batch_items_do_not_interact(model):
    glimpse, viewpoint = glimpse_batch(seed=1, batch_size=3)
    with torch.inference_mode():
        together = model(glimpse=glimpse, state=model.init_state(batch_size=3, canvas_grid_size=4), viewpoint=viewpoint)
        for i in range(3):
            alone = model(glimpse=glimpse[i : i + 1], state=model.init_state(batch_size=1, canvas_grid_size=4),
                          viewpoint=Viewpoint(centers=viewpoint.centers[i : i + 1], scales=viewpoint.scales[i : i + 1]))
            torch.testing.assert_close(together.state.canvas[i : i + 1], alone.state.canvas, atol=1e-4, rtol=1e-4)


@pytest.mark.slow
def test_torch_export_matches_eager(model, glimpses):
    glimpse, viewpoint = glimpses
    state = model.init_state(batch_size=2, canvas_grid_size=4)
    for cls in (Viewpoint, RecurrentState, CanViTOutput):
        torch.export.register_dataclass(cls, serialized_type_name=f"canvit_pytorch.{cls.__name__}")
    exported = torch.export.export(model, args=(), kwargs={"glimpse": glimpse, "state": state, "viewpoint": viewpoint})
    with torch.inference_mode():
        eager = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
    out = exported.module()(glimpse=glimpse, state=state, viewpoint=viewpoint)
    torch.testing.assert_close(out.state.canvas, eager.state.canvas, atol=1e-5, rtol=1e-5)
