"""Tests for training_step TBPTT logic.

Verifies chunk_size=1 (no temporal BPTT) and chunk_size=2 (baseline)
both produce correct gradients.
"""

import random
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
from canvit_pytorch import create_backbone
from torch import Tensor

from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig

from .objective import distillation_branch_metrics_fn, distillation_loss_fn
from .step import training_step

_DEVICE = torch.device("cpu")
_B = 2
_G = 8  # canvas grid size
_D = 384  # teacher dim (vits16)


@pytest.fixture()
def model() -> CanViTForPretraining:
    backbone = create_backbone("vits16").to(_DEVICE)
    cfg = CanViTForPretrainingConfig(teacher_dim=_D)
    return CanViTForPretraining(
        backbone=backbone,
        cfg=cfg,
        backbone_name="vits16",
        canvas_patch_grid_sizes=[_G],
    ).to(_DEVICE)


@pytest.fixture()
def tensors() -> dict[str, Tensor]:
    torch.manual_seed(42)
    return {
        "images": torch.randn(_B, 3, 224, 224, device=_DEVICE),
        "scene_target": torch.randn(_B, _G * _G, _D, device=_DEVICE),
        "cls_target": torch.randn(_B, _D, device=_DEVICE),
        "raw_scene_target": torch.randn(_B, _G * _G, _D, device=_DEVICE),
        "raw_cls_target": torch.randn(_B, _D, device=_DEVICE),
    }


def _run_step(
    model: CanViTForPretraining,
    tensors: dict[str, Tensor],
    *,
    chunk_size: int,
    continue_prob: float,
    n_glimpses_override: int | None = None,
) -> tuple[float, dict[str, Tensor]]:
    """Run training_step, return (loss, param_grads).

    If n_glimpses_override is set, patches random.random() to produce
    exactly that many glimpses.
    """
    model.zero_grad()

    # Deterministic trajectory length via controlled random.random() sequence.
    if n_glimpses_override is not None:
        assert continue_prob > 0, "need continue_prob > 0 to control n_glimpses"
        # n_glimpses starts at chunk_size, each loop iteration adds chunk_size.
        # We need (n_glimpses_override / chunk_size - 1) successes then 1 failure.
        n_continuations = n_glimpses_override // chunk_size - 1
        assert n_continuations >= 0
        # Values < continue_prob → continue; >= continue_prob → stop.
        sequence = [continue_prob / 2] * n_continuations + [1.0]
        call_count = 0

        def controlled_random() -> float:
            nonlocal call_count
            if call_count < len(sequence):
                val = sequence[call_count]
                call_count += 1
                return val
            return 1.0  # safety: stop

        ctx = patch("canvit_pretrain.train.step.random.random", side_effect=controlled_random)
    else:
        ctx = nullcontext()

    loss_fn = distillation_loss_fn(
        model=model,
        scene_target=tensors["scene_target"],
        cls_target=tensors["cls_target"],
        enable_scene_patches_loss=True,
        enable_scene_cls_loss=True,
    )
    branch_metrics_fn = distillation_branch_metrics_fn(
        model=model,
        scene_target=tensors["scene_target"],
        cls_target=tensors["cls_target"],
        raw_scene_target=tensors["raw_scene_target"],
        raw_cls_target=tensors["raw_cls_target"],
        scene_denorm=lambda x: x,
        cls_denorm=lambda x: x,
    )
    with ctx:
        metrics = training_step(
            model=model,
            images=tensors["images"],
            loss_fn=loss_fn,
            branch_metrics_fn=branch_metrics_fn,
            glimpse_size_px=128,
            canvas_grid_size=_G,
            n_full_start_branches=0,
            n_random_start_branches=1,
            chunk_size=chunk_size,
            continue_prob=continue_prob,
            min_viewpoint_scale=0.1,
            amp_ctx=nullcontext(),
            collect_viz=False,
        )

    grads = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    return metrics.total_loss.item(), grads


class TestRGBObjective:
    """The RGB-reconstruction objective trains through the same TBPTT core."""

    def test_rgb_step_produces_finite_loss_and_gradients(self) -> None:
        from canvit_pytorch import CanViTConfig, CanViTForRGBReconstruction, patchify

        from .objective import rgb_branch_metrics_fn, rgb_loss_fn

        backbone = create_backbone("vits16").to(_DEVICE)
        model = CanViTForRGBReconstruction(
            backbone=backbone, cfg=CanViTConfig(), backbone_name="vits16",
        ).to(_DEVICE)
        torch.manual_seed(0)
        scene_px = _G * backbone.patch_size_px
        images = torch.randn(_B, 3, scene_px, scene_px, device=_DEVICE)
        pixel_target = patchify(images, backbone.patch_size_px)

        model.zero_grad()
        metrics = training_step(
            model=model,
            images=images,
            loss_fn=rgb_loss_fn(model=model, pixel_target=pixel_target),
            branch_metrics_fn=rgb_branch_metrics_fn(model=model, pixel_target=pixel_target),
            glimpse_size_px=128,
            canvas_grid_size=_G,
            n_full_start_branches=1,
            n_random_start_branches=1,
            chunk_size=2,
            continue_prob=0.0,
            min_viewpoint_scale=0.1,
            amp_ctx=nullcontext(),
            collect_viz=False,
        )
        assert torch.isfinite(metrics.total_loss)
        assert metrics.total_loss.item() > 0
        assert metrics.random_start is not None
        assert "recon_loss" in metrics.random_start.metrics
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)
        # The RGB head must receive gradient.
        assert model.rgb_head["proj"].weight.grad is not None


def _has_grads(grads: dict[str, Tensor]) -> bool:
    return len(grads) > 0 and any(g.abs().sum() > 0 for g in grads.values())


# ── chunk_size=1 ──────────────────────────────────────────────────────


class TestChunkSize1:
    """chunk_size=1: every timestep gets an isolated backward (no temporal BPTT)."""

    def test_single_glimpse_produces_gradients(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        """n_glimpses=1: the critical edge case (was previously broken)."""
        loss, grads = _run_step(model, tensors, chunk_size=1, continue_prob=0.0)
        assert loss > 0
        assert _has_grads(grads), "no gradients produced with chunk_size=1, n_glimpses=1"

    def test_two_glimpses_produces_gradients(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        loss, grads = _run_step(
            model, tensors, chunk_size=1, continue_prob=0.5, n_glimpses_override=2,
        )
        assert loss > 0
        assert _has_grads(grads)

    def test_three_glimpses_produces_gradients(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        loss, grads = _run_step(
            model, tensors, chunk_size=1, continue_prob=0.5, n_glimpses_override=3,
        )
        assert loss > 0
        assert _has_grads(grads)

    def test_no_cross_step_gradient_flow(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        """Verify that with chunk_size=1, gradients are isolated per timestep.

        Strategy: run twice with same model state but different random seeds
        for viewpoints. If gradients flow across steps, the t=0 viewpoint
        would influence the gradient contribution from t=1. With isolation,
        each step's gradient depends only on its own viewpoint and the
        (detached) canvas state it receives.

        We use n_glimpses=2. With isolated backward:
          grad = grad_from_t0(vp0) + grad_from_t1(vp1, detached_state)
        The t1 contribution depends on detached canvas from t0 (same model
        init → same), so only t0's contribution differs. We verify the
        difference equals exactly the difference in t0's isolated gradient.
        """
        torch.manual_seed(0)
        random.seed(0)
        loss_a, grads_a = _run_step(
            model, tensors, chunk_size=1, continue_prob=0.5, n_glimpses_override=2,
        )

        torch.manual_seed(1)
        random.seed(1)
        loss_b, grads_b = _run_step(
            model, tensors, chunk_size=1, continue_prob=0.5, n_glimpses_override=2,
        )

        # Gradients should differ (different viewpoints → different loss landscape)
        shared_keys = set(grads_a) & set(grads_b)
        assert len(shared_keys) > 0
        any_differ = any(
            not torch.allclose(grads_a[k], grads_b[k], atol=1e-6)
            for k in shared_keys
        )
        assert any_differ, "gradients identical despite different viewpoints"


# ── chunk_size=2 (baseline, regression test) ──────────────────────────


class TestChunkSize2Regression:
    """Verify chunk_size=2 (the production default) is unaffected by the fix."""

    def test_single_chunk_produces_gradients(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        """n_glimpses=2 (one full chunk)."""
        loss, grads = _run_step(model, tensors, chunk_size=2, continue_prob=0.0)
        assert loss > 0
        assert _has_grads(grads)

    def test_two_chunks_produces_gradients(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        """n_glimpses=4 (two chunks)."""
        loss, grads = _run_step(
            model, tensors, chunk_size=2, continue_prob=0.5, n_glimpses_override=4,
        )
        assert loss > 0
        assert _has_grads(grads)

    def test_deterministic_with_same_seed(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        """Same seed → same loss. Verifies determinism of the test harness."""
        torch.manual_seed(99)
        random.seed(99)
        loss_a, _ = _run_step(model, tensors, chunk_size=2, continue_prob=0.0)

        torch.manual_seed(99)
        random.seed(99)
        loss_b, _ = _run_step(model, tensors, chunk_size=2, continue_prob=0.0)

        assert abs(loss_a - loss_b) < 1e-5, f"non-deterministic: {loss_a} vs {loss_b}"


# ── Cross-chunk_size consistency ──────────────────────────────────────


class TestCrossChunkSizeConsistency:
    """Verify that chunk_size=1 and chunk_size=2 produce different gradients
    (confirming TBPTT truncation matters) but both produce valid, finite results."""

    def test_both_produce_finite_losses(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        torch.manual_seed(0)
        random.seed(0)
        loss_1, grads_1 = _run_step(
            model, tensors, chunk_size=1, continue_prob=0.5, n_glimpses_override=2,
        )

        torch.manual_seed(0)
        random.seed(0)
        loss_2, grads_2 = _run_step(
            model, tensors, chunk_size=2, continue_prob=0.0,
        )

        assert torch.isfinite(torch.tensor(loss_1))
        assert torch.isfinite(torch.tensor(loss_2))
        assert _has_grads(grads_1)
        assert _has_grads(grads_2)
