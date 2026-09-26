"""How closely CanViT predicts its DINOv3 teacher's features of the whole scene after every glimpse.

The metric of the paper's pretraining ablations (Appendix E): cosine similarity between the
teacher's features and CanViT's predictions of them, for the patch features decoded from the
canvas ("scene") and the CLS token decoded from the recurrent CLS token ("cls"). "norm" compares
in the teacher's per-position standardized space, where pretraining fits the predictions and
the paper reports; "raw" compares the destandardized predictions with the raw features.
Scenes are the ADE20K validation images, resized on the short side and center-cropped.

Saves {"per_timestep": [{"t", "scene_cos_raw", "cls_cos_raw", "scene_cos_norm", "cls_cos_norm"}, ...],
"n_images", "metadata"}.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from canvit_pytorch.benchmarks import ade20k
from canvit_pytorch.evaluate.config import EpisodeConfig
from canvit_pytorch.evaluate.tasks.task import Task
from canvit_pytorch.hub import repos
from canvit_pytorch.model import CanViTForPretraining
from canvit_pytorch.model.standardizer import PositionAwareStandardizer
from canvit_pytorch.policies import POLICIES
from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.teacher import TEACHER_REPO, load_teacher

log = logging.getLogger(__name__)

METRICS = ("scene_cos_raw", "cls_cos_raw", "scene_cos_norm", "cls_cos_norm")

ABLATION_EPISODE = EpisodeConfig(policy="random", num_glimpses=10)
"""The episodes of the paper's ablation study: 10 R-IID glimpses."""


class Scenes(Dataset[Tensor]):
    """The ADE20K validation images, without labels."""

    def __init__(self, *, root: Path, size_px: int) -> None:
        image_dir = root / "images" / "validation"
        self.paths = sorted(image_dir.glob("*.jpg"))
        assert self.paths, f"no images in {image_dir}"
        self.transform = preprocess(size_px)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        image = self.transform(Image.open(self.paths[index]).convert("RGB"))
        assert isinstance(image, Tensor)
        return image


def cosine_similarities(
    *, prediction: Tensor, raw_target: Tensor, standardizer: PositionAwareStandardizer,
) -> tuple[float, float]:
    """(norm, raw) mean cosine similarity of a standardized prediction [..., D] with a raw target [..., D]."""
    norm = F.cosine_similarity(prediction, standardizer(raw_target), dim=-1).mean().item()
    raw = F.cosine_similarity(standardizer.destandardize(prediction), raw_target, dim=-1).mean().item()
    return norm, raw


@dataclass(frozen=True, kw_only=True)
class Reconstruction(Task):
    """How closely CanViT predicts its DINOv3 teacher's patch and CLS features of the scene after every glimpse."""

    pretrained_repo: str = repos.FLAGSHIP
    episode: EpisodeConfig = ABLATION_EPISODE
    output: Path = Path("results/recon.pt")
    batch_size: int = 16
    num_workers: int = 4

    def __post_init__(self) -> None:
        spec = POLICIES[self.episode.policy]
        assert not spec.needs_segmentation_probe, f"{spec.paper_name} decodes the canvas with an ADE20K probe"

    @torch.inference_mode()
    def run(self) -> Path:
        device = self.torch_device
        root = ade20k.dataset_root()
        model = CanViTForPretraining.from_pretrained(self.pretrained_repo).to(device).eval()
        teacher = load_teacher(TEACHER_REPO, device)
        assert teacher.embed_dim == model.teacher_dim, (teacher.embed_dim, model.teacher_dim)
        assert self.episode.canvas_grid_size == model.teacher_patch_grid, (
            f"{self.pretrained_repo} predicts a {model.teacher_patch_grid}² teacher patch grid; "
            f"the canvas grid must match, got {self.episode.canvas_grid_size}"
        )
        dataset = Scenes(root=root, size_px=model.teacher_patch_grid * teacher.patch_size)
        sums = [dict.fromkeys(METRICS, 0.0) for _ in range(self.episode.num_glimpses)]
        start = time.monotonic()
        with self.autocast():
            for images in self.batches(dataset, description="Reconstruction"):
                images = images.to(device)
                with torch.autocast(device_type=device.type, enabled=False):
                    target = teacher(images)
                batch_size = images.shape[0]
                for step in self.episode.rollout(canvit=model.canvit, images=images):
                    scene_norm, scene_raw = cosine_similarities(
                        prediction=model.predict_teacher_patches(step.state.canvas), raw_target=target.patches.float(),
                        standardizer=model.teacher_patch_standardizer,
                    )
                    cls_norm, cls_raw = cosine_similarities(
                        prediction=model.predict_teacher_cls(step.state.recurrent_cls), raw_target=target.cls.float(),
                        standardizer=model.teacher_cls_standardizer,
                    )
                    for metric, value in zip(METRICS, (scene_raw, cls_raw, scene_norm, cls_norm), strict=True):
                        sums[step.t][metric] += value * batch_size
        per_timestep = [
            {"t": t, **{metric: total / len(dataset) for metric, total in totals.items()}}
            for t, totals in enumerate(sums)
        ]
        for row in per_timestep:
            log.info("t=%d scene_cos_norm=%.4f cls_cos_norm=%.4f", row["t"], row["scene_cos_norm"], row["cls_cos_norm"])
        return self.save(
            {"per_timestep": per_timestep, "n_images": len(dataset)}, wall_time_seconds=time.monotonic() - start,
        )
