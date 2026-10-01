"""Record a CanViT rollout with everything a renderer needs, glimpse by glimpse."""

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from canvit_pytorch.episode import run_episode
from canvit_pytorch.model.segmentation import CanViTForSemanticSegmentation
from canvit_pytorch.policies import Policy
from canvit_pytorch.preprocess import imagenet_denormalize
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint


@dataclass(frozen=True)
class GlimpseRecord:
    t: int
    viewpoint: Viewpoint  # one scene, on the CPU
    crop: NDArray[np.uint8]  # [glimpse_size_px, glimpse_size_px, 3]: what CanViT saw
    canvas: NDArray[np.float32]  # [G*G, canvas_dim]: canvas patches after this glimpse
    logits: NDArray[np.float32]  # [num_classes, G, G]: segmentation decoded from the canvas
    write_residuals: tuple[NDArray[np.float32], ...]
    """What each Canvas Attention Write added to the canvas patches, [G*G, canvas_dim] each; empty unless captured."""


@dataclass(frozen=True)
class Rollout:
    scene: NDArray[np.uint8]  # [H, W, 3], the image as CanViT saw it
    initial_canvas: NDArray[np.float32]  # [G*G, canvas_dim]; canvas values reach 10^4, beyond float16's precision
    annotation: NDArray[np.int64] | None  # [H, W] class indices of the scene, IGNORE_LABEL where unlabeled
    glimpses: tuple[GlimpseRecord, ...]
    canvas_grid_size: int
    glimpse_size_px: int


def _uint8_image(normalized: Tensor) -> NDArray[np.uint8]:
    """[3, H, W] ImageNet-normalized -> [H, W, 3] uint8."""
    return (imagenet_denormalize(normalized.float().cpu()).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


@torch.inference_mode()
def record(
    model: CanViTForSemanticSegmentation,
    image: Tensor,
    policy: Policy,
    *,
    num_glimpses: int,
    canvas_grid_size: int,
    glimpse_size_px: int,
    capture_writes: bool,
    annotation: Tensor | None,
) -> Rollout:
    """Run one rollout on a [1, 3, H, W] ImageNet-normalized image and record it, with the scene's [H, W] annotation."""
    assert image.ndim == 4 and image.shape[0] == 1, f"record takes one image, got {tuple(image.shape)}"
    canvit = model.canvit

    def canvas_patches(canvas: Tensor) -> NDArray[np.float32]:
        return canvit.canvas_patches(canvas)[0].float().cpu().numpy()

    residuals: list[Tensor] = []

    def keep_residual(_module: nn.Module, _inputs: object, residual: Tensor) -> None:
        residuals.append(residual)

    handles: list[RemovableHandle] = []
    if capture_writes:
        handles = [write.register_forward_hook(keep_residual) for write in canvit.canvas_writes]
    initial = canvit.init_state(batch_size=1, canvas_grid_size=canvas_grid_size)
    try:
        steps = run_episode(
            canvit=canvit, images=image, policy=policy, num_glimpses=num_glimpses,
            glimpse_size_px=glimpse_size_px, initial_state=initial,
        )
    finally:
        for handle in handles:
            handle.remove()
    num_writes = len(canvit.canvas_writes)
    assert len(residuals) == (num_writes * num_glimpses if capture_writes else 0)

    glimpses = []
    previous = canvas_patches(initial.canvas)
    for step in steps:
        canvas = canvas_patches(step.state.canvas)
        writes = tuple(canvas_patches(r) for r in residuals[step.t * num_writes : (step.t + 1) * num_writes])
        # Writes are the only change to the canvas patches: their residuals add up to the next canvas.
        assert not writes or np.allclose(previous + sum(writes), canvas, rtol=1e-4, atol=1e-3 * float(np.abs(canvas).max())), (
            f"glimpse {step.t}: the canvas before it plus its Write residuals is not the canvas after it, "
            f"max difference {float(np.abs(previous + sum(writes) - canvas).max())}")
        previous = canvas
        viewpoint = Viewpoint(centers=step.viewpoint.centers[:1].cpu(), scales=step.viewpoint.scales[:1].cpu())
        crop = sample_at_viewpoint(spatial=image, viewpoint=step.viewpoint, glimpse_size_px=glimpse_size_px)
        glimpses.append(GlimpseRecord(
            t=step.t,
            viewpoint=viewpoint,
            crop=_uint8_image(crop[0]),
            canvas=canvas,
            logits=model.logits(step.state.canvas)[0].cpu().numpy(),
            write_residuals=writes,
        ))
    return Rollout(
        scene=_uint8_image(image[0]),
        initial_canvas=canvas_patches(initial.canvas),
        annotation=None if annotation is None else annotation.cpu().numpy().astype(np.int64),
        glimpses=tuple(glimpses),
        canvas_grid_size=canvas_grid_size,
        glimpse_size_px=glimpse_size_px,
    )
