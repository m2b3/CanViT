"""Measure, for every ADE20K validation scene, how CanViT-B's prediction of its DINOv3 ViT-B features improves over
eight random glimpses (sequences.sweep_sequence_name: R-IID, seeded by the scene's number), where the glimpses leave
the scene unseen, and how clearly the teacher's PCA colors separate the annotated classes; rank.py ranks the table.

One row per scene. Columns per glimpse t (suffix _t): unseen_area (share of the scene's pixels inside none of glimpses
0..t); cos_all and cos_never (mean cosine similarity of predicted and teacher patch features, DINOv3 space, over all
patches and over the patches inside none of the eight glimpses); zcos_* (the same in the per-position standardized
space of the pretraining loss); rgb_* (mean distance between the predicted and teacher PCA colors, [0, 1]³ RGB). Per
scene: never_seen_patches, teacher_pca3 (variance share of three components), class_eta2, classes, separated_classes
(measures.class_separation), width and height (native pixels).
"""

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.benchmarks.ade20k import dataset_root
from canvit_pytorch.episode import run_episode
from canvit_pytorch.hub import repos
from canvit_pytorch.model.pretraining import CanViTForPretraining
from canvit_pytorch.policies import FixedSequence
from canvit_pytorch.teacher import TEACHER_REPO, load_teacher
from canvit_pytorch.viewpoint import Viewpoint
from PIL import Image

from experiments import logs
from experiments.ade20k import load, scene_categories, validation_ids
from experiments.distillation.measures import (
    class_separation,
    colors,
    explained_by_three_components,
    patch_cosines,
    teacher_pca,
)
from experiments.distillation.sequences import (
    GLIMPSE_PX,
    GRID,
    NUM_GLIMPSES,
    as_array,
    never_seen_patches,
    seen_pixels,
    sweep_sequence_name,
    viewpoint_sequence,
)
from experiments.outputs import WORK

log = logging.getLogger(__name__)

PER_GLIMPSE = ["unseen_area", "cos_all", "cos_never", "zcos_all", "zcos_never", "rgb_all", "rgb_never"]


@dataclass(frozen=True)
class Config:
    batch_size: int = 16
    limit: int | None = None
    """first N scenes only, for a smoke test"""
    out: Path = WORK / "distillation/sweep.csv"
    device: str = "mps"


def mean_over(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """[T, N] values -> [T] means over the masked patches, NaN when none is."""
    return values[:, mask].mean(axis=1) if mask.any() else np.full(values.shape[0], np.nan)


@torch.inference_mode()
def main(cfg: Config) -> None:
    image_ids = validation_ids()[:cfg.limit]
    categories = scene_categories()
    device = torch.device(cfg.device)
    model = CanViTForPretraining.from_pretrained(repos.FLAGSHIP).to(device).eval()
    assert model.teacher_patch_grid == GRID, model.teacher_patch_grid
    standardizer = model.teacher_patch_standardizer
    teacher = load_teacher(TEACHER_REPO, device)

    columns = ["image_id", "category", "width", "height", "sequence", "never_seen_patches", "teacher_pca3",
               "class_eta2", "classes", "separated_classes"]
    columns += [f"{name}_{t}" for name in PER_GLIMPSE for t in range(NUM_GLIMPSES)]
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    with cfg.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        start = time.perf_counter()
        for first in range(0, len(image_ids), cfg.batch_size):
            batch = image_ids[first:first + cfg.batch_size]
            scenes = [load(image_id) for image_id in batch]
            images = torch.stack([image for image, _ in scenes]).to(device)
            target = teacher(images).patches
            assert target.shape == (len(batch), GRID * GRID, teacher.embed_dim), target.shape
            sequences = [viewpoint_sequence(sweep_sequence_name(image_id), image_id, device) for image_id in batch]
            per_step = [Viewpoint(centers=torch.cat([s[t].centers for s in sequences]),
                                  scales=torch.cat([s[t].scales for s in sequences])) for t in range(NUM_GLIMPSES)]
            steps = run_episode(canvit=model.canvit, images=images, policy=FixedSequence(per_step),
                                num_glimpses=NUM_GLIMPSES, glimpse_size_px=GLIMPSE_PX,
                                initial_state=model.init_state(batch_size=len(batch), canvas_grid_size=GRID))
            predicted_z = torch.stack([model.predict_teacher_patches(step.state.canvas) for step in steps], dim=1)
            predicted = standardizer.destandardize(predicted_z).float().cpu().numpy()
            predicted_z = predicted_z.float().cpu().numpy()
            target_z = standardizer(target).float().cpu().numpy()
            target = target.float().cpu().numpy()
            for i, image_id in enumerate(batch):
                viewpoints = as_array(sequences[i])
                never = never_seen_patches(viewpoints)
                basis, limits = teacher_pca(target[i])
                teacher_colors = colors(target[i], basis, limits)
                rgb = np.stack([np.linalg.norm(colors(p, basis, limits) - teacher_colors, axis=-1) for p in predicted[i]])
                cos, zcos = patch_cosines(predicted[i], target[i]), patch_cosines(predicted_z[i], target_z[i])
                eta2, classes, separated = class_separation(teacher_colors, scenes[i][1])
                width, height = Image.open(dataset_root() / "images/validation" / f"{image_id}.jpg").size
                per_glimpse = {"unseen_area": 1 - seen_pixels(viewpoints).mean(axis=(1, 2)),
                               "cos_all": cos.mean(axis=1), "cos_never": mean_over(cos, never),
                               "zcos_all": zcos.mean(axis=1), "zcos_never": mean_over(zcos, never),
                               "rgb_all": rgb.mean(axis=1), "rgb_never": mean_over(rgb, never)}
                row = {"image_id": image_id, "category": categories[image_id], "width": width, "height": height,
                       "sequence": sweep_sequence_name(image_id), "never_seen_patches": int(never.sum()),
                       "teacher_pca3": round(explained_by_three_components(target[i]), 4),
                       "class_eta2": round(eta2, 4), "classes": classes, "separated_classes": separated}
                row |= {f"{name}_{t}": round(float(v), 4) for name, values in per_glimpse.items()
                        for t, v in enumerate(values)}
                writer.writerow(row)
            file.flush()
            log.info("%d/%d scenes, %.0f s; last %s cos_all %s cos_never %s", first + len(batch), len(image_ids),
                     time.perf_counter() - start, batch[-1], row["cos_all_0"], row[f"cos_never_{NUM_GLIMPSES - 1}"])
    log.info("wrote %s", cfg.out)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
