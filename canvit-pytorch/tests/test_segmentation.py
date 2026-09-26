"""Tests for SegmentationProbe + CanViTForSemanticSegmentation.

Most tests here are architecture sanity checks that run without network access.
The explicit ``network`` test covers published probe loading through
``from_pretrained``.
"""

import pytest
import torch

from canvit_pytorch import (
    CanViTForSemanticSegmentation,
    SegmentationProbe,
    Viewpoint,
    resolve_canvit_repo,
    sample_at_viewpoint,
)

NUM_CLASSES = 150
B = 2
CANVAS_GRID = 8  # small for fast CPU test


@pytest.fixture
def dummy_input():
    scene = torch.randn(B, 3, 128, 128)
    vp = Viewpoint.full_scene(batch_size=B, device=torch.device("cpu"))
    glimpse = sample_at_viewpoint(spatial=scene, viewpoint=vp, glimpse_size_px=128)
    return glimpse, vp


class TestSegmentationProbe:
    def test_forward_shape(self):
        D = 64
        H = W = 8
        probe = SegmentationProbe(embed_dim=D, num_classes=NUM_CLASSES, dropout=0.1, use_ln=True)
        x = torch.randn(B, H, W, D)
        out = probe(x)
        assert out.shape == (B, NUM_CLASSES, H, W)

    def test_predict_upsamples(self):
        D = 64
        probe = SegmentationProbe(embed_dim=D, num_classes=NUM_CLASSES).eval()
        x = torch.randn(B, 4, 4, D)
        out = probe.predict(x, target_size=(32, 32))
        assert out.shape == (B, NUM_CLASSES, 32, 32)

    def test_use_ln_false_uses_identity(self):
        import torch.nn as nn
        probe = SegmentationProbe(embed_dim=32, num_classes=10, use_ln=False)
        assert isinstance(probe.ln, nn.Identity)

    def test_state_dict_keys_stable(self):
        """Probe state_dict keys must NOT change without coordination —
        canvit-specialize' frozen-probe HF checkpoints depend on them, and so
        does CanViTForSemanticSegmentation when copying probe weights into
        its composed head submodule."""
        probe = SegmentationProbe(embed_dim=32, num_classes=10)
        keys = set(probe.state_dict().keys())
        assert keys == {
            "ln.weight", "ln.bias",
            "bn.weight", "bn.bias", "bn.running_mean", "bn.running_var", "bn.num_batches_tracked",
            "conv.weight", "conv.bias",
        }

    def test_state_dict_keys_no_ln(self):
        probe = SegmentationProbe(embed_dim=32, num_classes=10, use_ln=False)
        keys = set(probe.state_dict().keys())
        assert "ln.weight" not in keys and "ln.bias" not in keys

    def test_embed_dim_mismatch_raises(self):
        """Forward with the wrong feature dim must fail loudly, not silently."""
        probe = SegmentationProbe(embed_dim=768, num_classes=NUM_CLASSES)
        try:
            probe(torch.randn(1, 8, 8, 384))
        except AssertionError:
            return
        raise AssertionError("Expected AssertionError on embed_dim mismatch")

    def test_state_dict_roundtrip(self):
        """Save and reload state_dict — eval forward must match exactly."""
        probe = SegmentationProbe(embed_dim=64, num_classes=NUM_CLASSES, dropout=0.1).eval()
        probe2 = SegmentationProbe(embed_dim=64, num_classes=NUM_CLASSES, dropout=0.1).eval()
        probe2.load_state_dict(probe.state_dict())
        x = torch.randn(1, 8, 8, 64)
        with torch.inference_mode():
            assert torch.allclose(probe(x), probe2(x))


@pytest.mark.network
class TestSegmentationProbeHFIntegration:
    """Integration: load published probe from HuggingFace Hub."""

    def test_from_pretrained_published_canvas_probe(self) -> None:
        """The flagship 1024px/c64 canvas probe round-trips cleanly."""
        probe = SegmentationProbe.from_pretrained(resolve_canvit_repo("probe-ade20k-40k-s512-c32-in21k")).eval()
        assert probe.embed_dim == 1024
        assert probe.num_classes == 150
        with torch.inference_mode():
            out = probe(torch.randn(1, 32, 32, 1024))
        assert out.shape == (1, 150, 32, 32)


EXPECTED_CANVAS_DIM = 64  # = canvas_num_heads * canvas_head_dim (2 * 32)


def _minimal_seg_model() -> CanViTForSemanticSegmentation:
    """Construct a reduced CanViTForSemanticSegmentation for fast CPU tests.

    canvas_dim is computed (canvas_num_heads * canvas_head_dim, see CanViTConfig).
    """
    return CanViTForSemanticSegmentation(
        backbone_name="vits16",
        model_config={
            "n_canvas_registers": 4,
            "canvas_num_heads": 2,
            "canvas_head_dim": 32,  # 2*32 = 64 canvas_dim
            "rw_stride": 2,
            "canvas_update_mode": "additive",
            "enable_vpe": False,
        },
        num_classes=NUM_CLASSES,
        dropout=0.1,
        use_ln=True,
    )


class TestCanViTForSemanticSegmentation:
    def test_construction(self):
        seg = _minimal_seg_model()
        assert seg.num_classes == NUM_CLASSES
        assert seg.canvas_dim == EXPECTED_CANVAS_DIM
        assert isinstance(seg.head, SegmentationProbe)
        assert seg.head.embed_dim == EXPECTED_CANVAS_DIM

    def test_init_state(self):
        seg = _minimal_seg_model().eval()
        state = seg.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        # Canvas: [B, n_registers + G^2, canvas_dim]
        assert state.canvas.shape == (B, 4 + CANVAS_GRID ** 2, EXPECTED_CANVAS_DIM)

    def test_forward_shape(self, dummy_input):
        glimpse, vp = dummy_input
        seg = _minimal_seg_model().eval()
        state = seg.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        with torch.inference_mode():
            logits, new_state = seg(glimpse=glimpse, state=state, viewpoint=vp)
        assert logits.shape == (B, NUM_CLASSES, CANVAS_GRID, CANVAS_GRID)
        assert new_state.canvas.shape == state.canvas.shape

    def test_predict_upsamples_to_target_size(self, dummy_input):
        """predict() returns upsampled logits at the requested spatial resolution."""
        glimpse, vp = dummy_input
        seg = _minimal_seg_model().eval()
        state = seg.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        target = (64, 96)  # non-square, not a multiple of canvas_grid
        with torch.inference_mode():
            logits, new_state = seg.predict(glimpse=glimpse, state=state, viewpoint=vp, target_size=target)
        assert logits.shape == (B, NUM_CLASSES, *target)
        assert new_state.canvas.shape == state.canvas.shape

    def test_canvit_then_head(self, dummy_input):
        """Verify callers can separate CanViT + head steps: advance state via
        `self.canvit(...)`, then apply the head directly via `self.head(spatial)`."""
        glimpse, vp = dummy_input
        seg = _minimal_seg_model().eval()
        state = seg.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        with torch.inference_mode():
            out = seg.canvit(glimpse=glimpse, state=state, viewpoint=vp)
            spatial = seg.canvit.get_spatial(out.state.canvas).view(B, CANVAS_GRID, CANVAS_GRID, -1)
            logits = seg.head(spatial)
        assert logits.shape == (B, NUM_CLASSES, CANVAS_GRID, CANVAS_GRID)

    def test_state_dict_has_canvit_and_head(self):
        """State dict must namespace CanViT vs head so CanViTForSemanticSegmentation.
        from_pretrained can round-trip both halves cleanly."""
        seg = _minimal_seg_model()
        keys = list(seg.state_dict().keys())
        canvit_keys = [k for k in keys if k.startswith("canvit.")]
        head_keys = [k for k in keys if k.startswith("head.")]
        assert canvit_keys, "expected canvit.* keys"
        assert head_keys, "expected head.* keys"
        assert set(keys) == set(canvit_keys) | set(head_keys), (
            f"unexpected keys outside canvit/head: {set(keys) - set(canvit_keys) - set(head_keys)}"
        )

    def test_head_state_dict_matches_segmentation_probe_keys(self):
        """``self.head`` is a SegmentationProbe, so head.* keys must match a
        standalone SegmentationProbe — this is what makes the load path in
        ``from_pretrained_with_probe`` (head.load_state_dict(probe.state_dict()))
        work without remap."""
        seg = _minimal_seg_model()
        head_keys = {k.removeprefix("head.") for k in seg.state_dict() if k.startswith("head.")}
        probe = SegmentationProbe(embed_dim=EXPECTED_CANVAS_DIM, num_classes=NUM_CLASSES, dropout=0.1, use_ln=True)
        assert head_keys == set(probe.state_dict().keys())
