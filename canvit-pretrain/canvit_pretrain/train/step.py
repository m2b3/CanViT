"""Training step with truncated BPTT and independent branches.

Objective-agnostic: the per-timestep loss and the end-of-branch metrics are injected
(see :mod:`canvit_pretrain.train.objective`). The TBPTT / independent-branch control flow
here does not know whether it is distilling teacher features or reconstructing pixels.
"""

import random
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch
from canvit_pytorch import CanViT, CanViTOutput, RecurrentState, Viewpoint, sample_at_viewpoint
from torch import Tensor

from .viewpoint import Viewpoint as NamedViewpoint
from .viewpoint import ViewpointType
from .viz.image import imagenet_denormalize_to_numpy
from .viz.sample import VizSampleData, extract_sample0_viz


class LossOutput(NamedTuple):
    """Per-timestep loss: a backward target plus named scalar components for logging."""

    combined: Tensor  # summed active losses, carries grad
    components: dict[str, Tensor]  # named per-step scalar losses (e.g. scene_patches_loss)


class BranchMetrics(NamedTuple):
    """Aggregated metrics for one branch: the scalar loss + named EMA scalars."""

    loss: Tensor
    metrics: dict[str, Tensor]


# Per-timestep loss given the model output. Built by the objective (closes over targets/model).
LossFn = Callable[[CanViTOutput], LossOutput]
# End-of-branch metrics from the final recurrent state (e.g. teacher cosine sims).
BranchMetricsFn = Callable[[RecurrentState], dict[str, Tensor]]
# Maps final canvas -> a per-sample prediction for the viz panel (objective-specific).
VizPredictFn = Callable[[Tensor], Tensor]


@dataclass
class TrainVizData:
    """Viz data collected during one training branch (sample 0 only)."""

    image: np.ndarray  # [H, W, 3] denormalized input
    target_features: np.ndarray  # [G², D] reconstruction target (teacher features or pixels)
    viewpoints: list[NamedViewpoint]  # viewpoints used at each timestep
    viz_samples: list[VizSampleData] = field(default_factory=list)  # per-timestep
    initial_scene: np.ndarray | None = None  # [G², D] initial scene prediction
    initial_canvas_spatial: np.ndarray | None = None  # [G², C] initial canvas


class StepMetrics(NamedTuple):
    """Output from training_step."""

    total_loss: Tensor
    full_start: BranchMetrics | None  # None if n_full_start_branches=0
    random_start: BranchMetrics | None  # None if n_random_start_branches=0
    n_glimpses: int  # trajectory length this step
    viz_data: TrainVizData | None = None  # optional viz from first branch


@dataclass
class ChunkState:
    """State for TBPTT chunk processing."""

    state: RecurrentState
    vpe: Tensor | None
    chunk_combined_loss: Tensor  # with grad
    total_combined_loss: Tensor  # detached
    component_totals: dict[str, Tensor]  # detached, per named component
    n_steps: int


class StepOutput(NamedTuple):
    """Output from forward_glimpse: model output + sampled glimpse."""

    out: CanViTOutput
    glimpse: Tensor


def training_step(
    *,
    model: CanViT,
    images: Tensor,
    loss_fn: LossFn,
    branch_metrics_fn: BranchMetricsFn,
    glimpse_size_px: int,
    canvas_grid_size: int,
    n_full_start_branches: int,
    n_random_start_branches: int,
    chunk_size: int,
    continue_prob: float,
    min_viewpoint_scale: float,
    amp_ctx: AbstractContextManager,
    collect_viz: bool = False,
    viz_predict_fn: VizPredictFn | None = None,
    viz_target: Tensor | None = None,
) -> StepMetrics:
    """Training with truncated BPTT and independent branches.

    Each branch is fully independent: own t0, own trajectory, own backward.
    No retain_graph needed. Memory is O(chunk_size), not O(n_branches).

    ``loss_fn`` and ``branch_metrics_fn`` are supplied by the objective. Viz (sample-0
    PCA panels) is teacher/RGB-specific via ``viz_predict_fn`` + ``viz_target`` and only
    runs when ``collect_viz`` is set; callers that don't want viz pass ``collect_viz=False``.
    """
    n_branches = n_full_start_branches + n_random_start_branches
    assert n_branches >= 1
    assert chunk_size >= 1
    assert 0.0 <= continue_prob <= 1.0
    if collect_viz:
        assert viz_predict_fn is not None and viz_target is not None, "viz needs viz_predict_fn + viz_target"
    device = images.device
    B = images.shape[0]

    state_init = model.init_state(batch_size=B, canvas_grid_size=canvas_grid_size)

    # Sample trajectory length (shared across branches for this step)
    n_glimpses = chunk_size
    while random.random() < continue_prob:
        n_glimpses += chunk_size

    # t1_schedule[t-1][branch_idx] = viewpoint type for timestep t, branch branch_idx
    # t>=1 is always all-RANDOM
    t1_schedule: list[list[ViewpointType]] = [
        [ViewpointType.RANDOM] * n_branches for _ in range(1, n_glimpses)
    ]

    full_metrics: list[BranchMetrics] = []
    random_metrics: list[BranchMetrics] = []

    # Viz collection for first branch only (when enabled)
    viz_data: TrainVizData | None = None

    def make_named_vp(vp_type: ViewpointType) -> NamedViewpoint:
        """Create a NamedViewpoint (has .name for viz, convertible to canvit Viewpoint)."""
        if vp_type == ViewpointType.RANDOM:
            return NamedViewpoint.random(batch_size=B, device=device, min_scale=min_viewpoint_scale)
        assert vp_type == ViewpointType.FULL
        return NamedViewpoint.full_scene(batch_size=B, device=device)

    def to_canvit_vp(vp: NamedViewpoint) -> Viewpoint:
        return Viewpoint(centers=vp.centers, scales=vp.scales)

    def forward_glimpse(*, state: RecurrentState, vp: Viewpoint) -> StepOutput:
        glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=glimpse_size_px)
        out = model(glimpse=glimpse, state=state, viewpoint=vp)
        return StepOutput(out=out, glimpse=glimpse)

    def record_viz_sample(out: CanViTOutput, glimpse: Tensor) -> None:
        assert viz_data is not None and viz_predict_fn is not None
        pred = viz_predict_fn(out.state.canvas)
        viz_data.viz_samples.append(extract_sample0_viz(out, glimpse, pred, model))

    def run_branch(t0_type: ViewpointType, branch_idx: int) -> BranchMetrics:
        nonlocal viz_data
        do_viz = collect_viz and branch_idx == 0

        # Capture initial state for viz (before any glimpses)
        if do_viz:
            assert viz_predict_fn is not None and viz_target is not None
            init_scene = viz_predict_fn(state_init.canvas)
            init_spatial = model.get_spatial(state_init.canvas[0:1])[0]
            viz_data = TrainVizData(
                image=imagenet_denormalize_to_numpy(images[0]),
                target_features=viz_target[0].cpu().float().numpy(),
                viewpoints=[],
                viz_samples=[],
                initial_scene=init_scene[0].detach().cpu().float().numpy(),
                initial_canvas_spatial=init_spatial.detach().cpu().float().numpy(),
            )

        # t0 forward
        with amp_ctx:
            vp0_named = make_named_vp(t0_type)
            vp0 = to_canvit_vp(vp0_named)
            step_out = forward_glimpse(state=state_init, vp=vp0)
            out, glimpse = step_out.out, step_out.glimpse
            L = loss_fn(out)

        if do_viz:
            assert viz_data is not None
            viz_data.viewpoints.append(vp0_named)
            record_viz_sample(out, glimpse)

        chunk = ChunkState(
            state=out.state,
            vpe=out.vpe,
            chunk_combined_loss=L.combined.float(),
            total_combined_loss=L.combined.detach().float(),
            component_totals={k: v.detach().float() for k, v in L.components.items()},
            n_steps=1,
        )

        # t=0 constitutes a complete chunk when chunk_size=1.
        if chunk_size == 1:
            loss_for_backward = chunk.chunk_combined_loss / n_glimpses / n_branches
            loss_for_backward.backward()
            if n_glimpses > 1:
                chunk.state = RecurrentState(
                    canvas=out.state.canvas.detach(),
                    recurrent_cls=out.state.recurrent_cls.detach(),
                )
                chunk.vpe = out.vpe.detach() if out.vpe is not None else None
                chunk.chunk_combined_loss = torch.zeros((), device=device)

        for t in range(1, n_glimpses):
            # t>=1: use pre-computed schedule (half RANDOM, half POLICY, shuffled)
            vp_type = t1_schedule[t - 1][branch_idx]
            vp_named = make_named_vp(vp_type)
            vp = to_canvit_vp(vp_named)

            with amp_ctx:
                step_out = forward_glimpse(state=chunk.state, vp=vp)
                out, glimpse = step_out.out, step_out.glimpse
                L = loss_fn(out)

            if do_viz:
                assert viz_data is not None
                viz_data.viewpoints.append(vp_named)
                record_viz_sample(out, glimpse)

            chunk.chunk_combined_loss = chunk.chunk_combined_loss + L.combined.float()
            chunk.total_combined_loss = chunk.total_combined_loss + L.combined.detach().float()
            for k, v in L.components.items():
                chunk.component_totals[k] = chunk.component_totals[k] + v.detach().float()
            chunk.n_steps += 1

            is_chunk_end = ((t + 1) % chunk_size == 0)
            is_last = (t == n_glimpses - 1)

            if is_chunk_end:
                loss_for_backward = chunk.chunk_combined_loss / n_glimpses / n_branches
                loss_for_backward.backward()  # no retain_graph

                if not is_last:
                    chunk.state = RecurrentState(
                        canvas=out.state.canvas.detach(),
                        recurrent_cls=out.state.recurrent_cls.detach(),
                    )
                    chunk.vpe = out.vpe.detach() if out.vpe is not None else None
                    chunk.chunk_combined_loss = torch.zeros((), device=device)
                else:
                    chunk.state = out.state
                    chunk.vpe = out.vpe
            else:
                chunk.state = out.state
                chunk.vpe = out.vpe

        n = chunk.n_steps
        metrics: dict[str, Tensor] = {k: v / n for k, v in chunk.component_totals.items()}
        metrics.update(branch_metrics_fn(chunk.state))
        return BranchMetrics(loss=chunk.total_combined_loss / n, metrics=metrics)

    # Run all branches (full-start first, then random-start)
    branch_idx = 0
    for _ in range(n_full_start_branches):
        full_metrics.append(run_branch(ViewpointType.FULL, branch_idx))
        branch_idx += 1

    for _ in range(n_random_start_branches):
        random_metrics.append(run_branch(ViewpointType.RANDOM, branch_idx))
        branch_idx += 1

    def aggregate(metrics: list[BranchMetrics]) -> BranchMetrics | None:
        if not metrics:
            return None
        keys = metrics[0].metrics.keys()
        return BranchMetrics(
            loss=torch.stack([m.loss for m in metrics]).mean(),
            metrics={k: torch.stack([m.metrics[k] for m in metrics]).mean() for k in keys},
        )

    full_start = aggregate(full_metrics)
    random_start = aggregate(random_metrics)

    all_losses = [m.loss for m in full_metrics] + [m.loss for m in random_metrics]
    total_loss = torch.stack(all_losses).mean()

    return StepMetrics(
        total_loss=total_loss,
        full_start=full_start,
        random_start=random_start,
        n_glimpses=n_glimpses,
        viz_data=viz_data,
    )
