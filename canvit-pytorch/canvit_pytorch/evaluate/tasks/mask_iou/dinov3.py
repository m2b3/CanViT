import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from canvit_pytorch.evaluate.config import SCENE_SIZE_PX
from canvit_pytorch.evaluate.tasks.ade20k_segmentation import ade20k_validation_set, dinov3_patch_grid, probe_logits
from canvit_pytorch.evaluate.tasks.mask_iou import table
from canvit_pytorch.evaluate.tasks.task import Task
from canvit_pytorch.hub import repos
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.teacher import DINOV3_REPOS, DINOv3Variant, load_teacher

log = logging.getLogger(__name__)

VARIANT: DINOv3Variant = "vitb16"
"""CanViT-B's teacher."""


def probe_repo(input_size_px: int) -> str:
    return repos.released_dinov3_ade20k_probe(repos.DINOV3_PROBE_MODEL_NAMES[VARIANT], input_size_px=input_size_px)


@dataclass(frozen=True, kw_only=True)
class MaskIoUDINOv3(Task):
    """Per-mask ADE20K IoU of CanViT-B's teacher, DINOv3 ViT-B/16, with its linear probe at each input size."""

    input_sizes_px: tuple[int, ...] = (128,)
    scene_size_px: int = SCENE_SIZE_PX
    output: Path = Path("results/ade20k_obj/dv3_iou.parquet")
    batch_size: int = 32

    @torch.inference_mode()
    def run(self) -> Path:
        device = self.torch_device
        dataset = ade20k_validation_set(self.scene_size_px)
        table.check_image_count(len(dataset))
        teacher = load_teacher(DINOV3_REPOS[VARIANT], device)
        frames: list[pd.DataFrame] = []
        for input_size_px in self.input_sizes_px:
            probe = SegmentationProbe.from_pretrained(probe_repo(input_size_px)).to(device).eval()
            counts: list[torch.Tensor] = []
            for images, labels in self.batches(dataset, description=f"DINOv3 at {input_size_px} px"):
                with self.autocast():
                    features = dinov3_patch_grid(teacher, images.to(device), input_size_px=input_size_px)
                counts.append(table.per_image_counts(probe_logits(probe, features), labels.to(device)))
            intersection, union, label_area = torch.cat(counts, dim=1)
            table.log_mean_iou(intersection=intersection, union=union, label=f"DINOv3 at {input_size_px} px")
            frames.append(table.per_mask_rows(
                intersection=intersection, union=union, label_area=label_area,
                mask_resolution_px=self.scene_size_px, resolution=input_size_px,
            ))
        table.write(pd.concat(frames, ignore_index=True), output=self.output, runs=self.metadata(
            dinov3_repo=DINOV3_REPOS[VARIANT], probe_repos={size: probe_repo(size) for size in self.input_sizes_px},
        ))
        return self.output
