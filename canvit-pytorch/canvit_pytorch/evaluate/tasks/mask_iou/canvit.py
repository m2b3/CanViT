import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from canvit_pytorch.benchmarks import ade20k
from canvit_pytorch.evaluate.config import GLIMPSE_SIZE_PX, NUM_GLIMPSES, SCENE_SIZE_PX, EpisodeConfig
from canvit_pytorch.evaluate.tasks.ade20k_segmentation import ade20k_validation_set
from canvit_pytorch.evaluate.tasks.mask_iou import table
from canvit_pytorch.evaluate.tasks.task import Task
from canvit_pytorch.hub import repos
from canvit_pytorch.model import CanViTForSemanticSegmentation
from canvit_pytorch.policies import PolicyName

log = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class MaskIoUCanViT(Task):
    """Per-mask ADE20K IoU of CanViT after every glimpse, at each canvas grid size with its own linear probe.

    The table at output keeps the rows of canvas grids this run does not evaluate.
    """

    canvas_grid_sizes: tuple[int, ...] = (8, 16, 32, 64)
    pretraining_dataset: repos.PretrainingDataset = "in21k"
    """Which pretrained CanViT-B, with the ADE20K probes trained on its canvas."""
    policy: PolicyName = "entropy_coarse_to_fine"
    num_glimpses: int = NUM_GLIMPSES
    glimpse_size_px: int = GLIMPSE_SIZE_PX
    scene_size_px: int = SCENE_SIZE_PX
    output: Path = Path("results/ade20k_obj/canvit_iou.parquet")
    batch_size: int = 8

    def __post_init__(self) -> None:
        for canvas_grid_size in self.canvas_grid_sizes:
            self.episode(canvas_grid_size)  # checks that the policy supports the canvas grid

    @property
    def pretrained_repo(self) -> str:
        return repos.PRETRAINED[self.pretraining_dataset]

    def probe_repo(self, canvas_grid_size: int) -> str:
        return repos.released_ade20k_probe(
            self.pretraining_dataset, scene_size_px=self.scene_size_px, canvas_grid_size=canvas_grid_size,
        )

    def episode(self, canvas_grid_size: int) -> EpisodeConfig:
        return EpisodeConfig(
            policy=self.policy, num_glimpses=self.num_glimpses,
            canvas_grid_size=canvas_grid_size, glimpse_size_px=self.glimpse_size_px,
        )

    def run(self) -> Path:
        dataset = ade20k_validation_set(self.scene_size_px)
        table.check_image_count(len(dataset))
        for canvas_grid_size in self.canvas_grid_sizes:
            rows = self.evaluate(canvas_grid_size, dataset)
            table.replace_canvas_grid(rows, output=self.output, canvas_grid_size=canvas_grid_size, run=self.metadata(
                canvas_grid_size=canvas_grid_size, pretrained_repo=self.pretrained_repo,
                probe_repo=self.probe_repo(canvas_grid_size),
            ))
        return self.output

    @torch.inference_mode()
    def evaluate(self, canvas_grid_size: int, dataset: ade20k.ADE20kDataset) -> pd.DataFrame:
        device = self.torch_device
        seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(
            pretrained_repo=self.pretrained_repo, probe_repo=self.probe_repo(canvas_grid_size),
        ).to(device).eval()
        episode = self.episode(canvas_grid_size)
        counts: list[torch.Tensor] = []  # [T, 3, B, num_classes] per batch
        with self.autocast():
            for images, labels in self.batches(dataset, description=f"CanViT, {canvas_grid_size}² canvas"):
                steps = episode.rollout(canvit=seg.canvit, images=images.to(device), canvas_logits=seg.logits)
                labels = labels.to(device)
                counts.append(torch.stack([
                    table.per_image_counts(seg.logits(step.state.canvas), labels) for step in steps
                ]))
        intersection, union, label_area = torch.cat(counts, dim=2).unbind(dim=1)  # [T, N, num_classes] each
        for t in sorted({0, self.num_glimpses - 1}):
            table.log_mean_iou(intersection=intersection[t], union=union[t], label=f"{canvas_grid_size}² canvas, t={t}")
        return pd.concat([
            table.per_mask_rows(
                intersection=intersection[t], union=union[t], label_area=label_area[0],  # the same after every glimpse
                mask_resolution_px=self.scene_size_px, canvas_resolution=canvas_grid_size, timestep=t,
            )
            for t in range(self.num_glimpses)
        ], ignore_index=True)
