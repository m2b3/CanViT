"""Save, uncolored, what the distillation slide needs for chosen ADE20K validation scenes: DINOv3 ViT-B's patch features
of the whole 512 px scene (the target), its ADE20K probe's logits, and, for named glimpse sequences (sequences.py), the
pretrained CanViT-B's prediction of those features (destandardized) after every glimpse. plot.py draws them."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.episode import run_episode
from canvit_pytorch.hub import repos
from canvit_pytorch.model.pretraining import CanViTForPretraining
from canvit_pytorch.policies import FixedSequence
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.teacher import TEACHER_REPO, load_teacher
from canvit_pytorch.viewpoint import sample_at_viewpoint

from experiments import logs
from experiments.ade20k import load, pixels
from experiments.distillation.measures import patch_cosines
from experiments.distillation.sequences import (
    GLIMPSE_PX,
    GRID,
    SCENE_PX,
    as_array,
    never_seen_patches,
    sweep_sequence_name,
    viewpoint_sequence,
)
from experiments.outputs import WORK

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    ids: tuple[str, ...]
    """ADE20K validation image ids"""
    sequences: tuple[str, ...] = ()
    """sequence names (sequences.py); each scene's sweep sequence when empty"""
    out: Path = WORK / "distillation/exports"
    device: str = "mps"


@torch.inference_mode()
def main(cfg: Config) -> None:
    device = torch.device(cfg.device)
    model = CanViTForPretraining.from_pretrained(repos.FLAGSHIP).to(device).eval()
    assert model.teacher_patch_grid == GRID, model.teacher_patch_grid
    teacher = load_teacher(TEACHER_REPO, device)
    probe = SegmentationProbe.from_pretrained(
        repos.released_dinov3_ade20k_probe("dv3b", input_size_px=SCENE_PX)).to(device).eval()
    cfg.out.mkdir(parents=True, exist_ok=True)
    for image_id in cfg.ids:
        image, labels = load(image_id)
        images = image[None].to(device)
        target = teacher(images)
        assert target.patches.shape == (1, GRID * GRID, teacher.embed_dim), target.patches.shape
        teacher_patches = target.patches[0].float().cpu().numpy()
        arrays: dict[str, np.ndarray] = {
            "scene": pixels(image),
            "labels": labels.astype(np.uint8),
            "teacher_patches": teacher_patches.astype(np.float16),
            "teacher_logits": probe(target.patches.view(1, GRID, GRID, -1).float())[0].cpu().numpy().astype(np.float16),
        }
        names = list(cfg.sequences) or [sweep_sequence_name(image_id)]
        for name in names:
            viewpoints = viewpoint_sequence(name, image_id, device)
            steps = run_episode(canvit=model.canvit, images=images, policy=FixedSequence(viewpoints),
                                num_glimpses=len(viewpoints), glimpse_size_px=GLIMPSE_PX,
                                initial_state=model.init_state(batch_size=1, canvas_grid_size=GRID))
            predicted = torch.stack([
                model.teacher_patch_standardizer.destandardize(model.predict_teacher_patches(step.state.canvas))[0]
                for step in steps]).float().cpu().numpy()
            arrays[f"{name}/viewpoints"] = as_array(viewpoints)
            arrays[f"{name}/glimpses"] = np.stack([
                pixels(sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=GLIMPSE_PX)[0])
                for vp in viewpoints])
            arrays[f"{name}/predicted_patches"] = predicted.astype(np.float16)
            cos = patch_cosines(predicted, teacher_patches)
            never = never_seen_patches(arrays[f"{name}/viewpoints"])
            log.info("%s %s: mean patch cosine to the teacher by glimpse, all patches %s, the %d never inside a glimpse "
                     "%s", image_id, name, cos.mean(1).round(3).tolist(), never.sum(),
                     cos[:, never].mean(1).round(3).tolist() if never.any() else None)
        meta = {"image_id": image_id, "scene_px": SCENE_PX, "glimpse_px": GLIMPSE_PX, "grid": GRID, "sequences": names,
                "model": repos.FLAGSHIP, "teacher": TEACHER_REPO}
        path = cfg.out / f"{image_id}.npz"
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
        log.info("%s", path)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
