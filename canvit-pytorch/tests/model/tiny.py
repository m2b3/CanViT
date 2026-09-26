import torch

from canvit_pytorch import CanViTConfig, Viewpoint, sample_at_viewpoint

# A small CanViT: ViT-S/16 backbone, 64-dim canvas. Paper-sized models are exercised through the released checkpoints.
TINY = CanViTConfig(backbone_name="vits16", canvas_num_heads=2, canvas_head_dim=32, num_canvas_registers=4,
                    num_backbone_registers=2, rw_stride=2)
GLIMPSE_PX = 32


def glimpse_batch(*, seed: int, batch_size: int) -> tuple[torch.Tensor, Viewpoint]:
    """Random scenes, each seen through a different viewpoint."""
    generator = torch.Generator().manual_seed(seed)
    scenes = torch.randn(batch_size, 3, 64, 64, generator=generator)
    viewpoint = Viewpoint(centers=torch.rand(batch_size, 2, generator=generator) - 0.5,
                          scales=0.3 + 0.5 * torch.rand(batch_size, generator=generator))
    return sample_at_viewpoint(spatial=scenes, viewpoint=viewpoint, glimpse_size_px=GLIMPSE_PX), viewpoint

