"""DINOv3 ViT-B on the whole scene at a glimpse's resolution, decoded by its released ADE20K probe for that input size:
the passive model given the pixels CanViT gets from one glimpse."""

import numpy as np
import torch
from canvit_pytorch.hub.repos import released_dinov3_ade20k_probe
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.teacher import TEACHER_REPO, load_teacher
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint

from experiments.glimpses import GLIMPSE_PX


class WholeSceneDINOv3:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.teacher = load_teacher(TEACHER_REPO, device)
        self.probe_repo = released_dinov3_ade20k_probe("dv3b", input_size_px=GLIMPSE_PX)
        self.probe = SegmentationProbe.from_pretrained(self.probe_repo).to(device).eval()
        self.grid = GLIMPSE_PX // self.teacher.patch_size

    @torch.inference_mode()
    def logits(self, image: torch.Tensor) -> np.ndarray:
        """An ImageNet-normalized [3, S, S] scene -> [num_classes, grid, grid] logits, one cell per DINOv3 patch."""
        whole = Viewpoint.full_scene(batch_size=1, device=self.device)
        crop = sample_at_viewpoint(spatial=image[None].to(self.device), viewpoint=whole, glimpse_size_px=GLIMPSE_PX)
        patches = self.teacher(crop).patches.view(1, self.grid, self.grid, -1).float()
        return self.probe(patches)[0].cpu().numpy()
