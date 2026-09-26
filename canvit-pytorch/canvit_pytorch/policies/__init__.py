"""Viewing policies: where CanViT looks next (paper, Section 6 and Appendix D.1).

POLICIES holds every policy the paper evaluates with the facts other code
needs: its name in the paper, a description, whether it is deterministic, and
whether it decodes the canvas with a segmentation probe.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

from canvit_pytorch.model.canvit import RecurrentState
from canvit_pytorch.policies.entropy import CanvasLogits, EntropyGuidedC2F, canvas_grid_supported
from canvit_pytorch.policies.quadtree import coarse_to_fine, fine_to_coarse
from canvit_pytorch.policies.random import MIN_SCALE, random_viewpoints
from canvit_pytorch.viewpoint import Viewpoint

PolicyName = Literal[
    "full_then_random", "random", "coarse_to_fine", "fine_to_coarse", "entropy_coarse_to_fine", "repeated_full_scene",
]


class Policy(Protocol):
    def step(self, t: int, state: RecurrentState) -> Viewpoint:
        """The viewpoint of glimpse t, given the state before it."""
        ...


@dataclass(frozen=True)
class FixedSequence:
    """A policy whose viewpoints are drawn before the rollout starts."""

    viewpoints: list[Viewpoint]

    def step(self, t: int, state: RecurrentState) -> Viewpoint:
        return self.viewpoints[t]


@dataclass(frozen=True)
class PolicyContext:
    batch_size: int
    device: torch.device
    num_glimpses: int
    canvas_grid_size: int
    canvas_logits: CanvasLogits | None


@dataclass(frozen=True)
class PolicySpec:
    paper_name: str
    description: str
    deterministic: bool
    needs_segmentation_probe: bool
    build: Callable[[PolicyContext], Policy]

    def supports_canvas_grid(self, canvas_grid_size: int) -> bool:
        return not self.needs_segmentation_probe or canvas_grid_supported(canvas_grid_size)


def _random_sequence(ctx: PolicyContext, *, start_with_full_scene: bool) -> FixedSequence:
    first = Viewpoint.full_scene(batch_size=ctx.batch_size, device=ctx.device) if start_with_full_scene else None
    rest = [
        random_viewpoints(batch_size=ctx.batch_size, device=ctx.device, min_scale=MIN_SCALE)
        for _ in range(ctx.num_glimpses - (first is not None))
    ]
    return FixedSequence(([first] if first is not None else []) + rest)


def _entropy_guided(ctx: PolicyContext) -> EntropyGuidedC2F:
    assert ctx.canvas_logits is not None, "EG-C2F decodes the canvas: pass canvas_logits"
    return EntropyGuidedC2F(
        batch_size=ctx.batch_size, device=ctx.device, num_glimpses=ctx.num_glimpses,
        canvas_grid_size=ctx.canvas_grid_size, canvas_logits=ctx.canvas_logits,
    )


POLICIES: dict[PolicyName, PolicySpec] = {
    "full_then_random": PolicySpec(
        paper_name="F-IID",
        description="The full scene first, then random viewpoints; one of the two pretraining rollouts.",
        deterministic=False, needs_segmentation_probe=False,
        build=lambda ctx: _random_sequence(ctx, start_with_full_scene=True),
    ),
    "random": PolicySpec(
        paper_name="R-IID",
        description="Random viewpoints from the first glimpse on; the other pretraining rollout.",
        deterministic=False, needs_segmentation_probe=False,
        build=lambda ctx: _random_sequence(ctx, start_with_full_scene=False),
    ),
    "coarse_to_fine": PolicySpec(
        paper_name="C2F",
        description="A quadtree from coarse to fine: the full scene, then its quadrants, then sixteenths, "
                    "each level in random order.",
        deterministic=False, needs_segmentation_probe=False,
        build=lambda ctx: FixedSequence(coarse_to_fine(batch_size=ctx.batch_size, device=ctx.device, num_glimpses=ctx.num_glimpses)),
    ),
    "fine_to_coarse": PolicySpec(
        paper_name="F2C",
        description="C2F's viewpoints from the finest level to the coarsest, isolating the effect of order.",
        deterministic=False, needs_segmentation_probe=False,
        build=lambda ctx: FixedSequence(fine_to_coarse(batch_size=ctx.batch_size, device=ctx.device, num_glimpses=ctx.num_glimpses)),
    ),
    "entropy_coarse_to_fine": PolicySpec(
        paper_name="EG-C2F",
        description="C2F's levels, each visiting next the tile where the segmentation decoded from the canvas "
                    "is least certain.",
        deterministic=True, needs_segmentation_probe=True,
        build=_entropy_guided,
    ),
    "repeated_full_scene": PolicySpec(
        paper_name="RFS",
        description="The full scene at every glimpse: recurrence without new input.",
        deterministic=True, needs_segmentation_probe=False,
        build=lambda ctx: FixedSequence([Viewpoint.full_scene(batch_size=ctx.batch_size, device=ctx.device)] * ctx.num_glimpses),
    ),
}


def make_policy(
    name: PolicyName, *, batch_size: int, device: torch.device, num_glimpses: int, canvas_grid_size: int,
    canvas_logits: CanvasLogits | None = None,
) -> Policy:
    spec = POLICIES[name]
    assert spec.supports_canvas_grid(canvas_grid_size), f"{spec.paper_name} does not support a {canvas_grid_size}² canvas"
    return spec.build(PolicyContext(batch_size, device, num_glimpses, canvas_grid_size, canvas_logits))


__all__ = ["MIN_SCALE", "POLICIES", "FixedSequence", "Policy", "PolicyName", "PolicySpec", "make_policy", "random_viewpoints"]
