"""A scene and an R-IID seed for "CanViT in action" whose rollout's mIoU on the scene never drops from one glimpse to
the next. Each seed's viewpoints are drawn exactly as `canvit_pytorch.viz rollout --policy random --seed S` draws them,
so the chosen rollout records as a bundle with that command; the mIoU is canvit_pytorch.viz.web.mean_iou, the bundles'
own, computed in batches and checked against it. Writes one JSON line per scene: each seed's mIoU after every glimpse."""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from canvit_pytorch.benchmarks.ade20k import IGNORE_LABEL
from canvit_pytorch.policies import FixedSequence, make_policy
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE, GLIMPSE_SIZE_PX, load_released_segmenter
from canvit_pytorch.viz.web import mean_iou

from experiments import logs
from experiments.ade20k import SCENE_PX, load, validation_ids
from experiments.outputs import WORK

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    scenes: tuple[str, ...] = ()
    """ADE20K validation image ids, in addition to the --sample."""
    sample: int = 0
    sample_seed: int = 0
    seeds: int = 32
    """R-IID seeds 0 .. seeds - 1 per scene, run as one batch."""
    glimpses: int = 21
    out: Path = WORK / "rollout_demo/sweep.jsonl"
    device: str = "mps"


def seed_viewpoints(seed: int, glimpses: int) -> list[Viewpoint]:
    """The viewpoints `canvit_pytorch.viz rollout --policy random --seed seed` takes, on the CPU as it draws them."""
    torch.manual_seed(seed)
    policy = make_policy("random", batch_size=1, device=torch.device("cpu"), num_glimpses=glimpses,
                         canvas_grid_size=CANVAS_GRID_SIZE)
    assert isinstance(policy, FixedSequence)
    return policy.viewpoints


def batched_miou(logits: torch.Tensor, truth: torch.Tensor, num_classes: int) -> list[float]:
    """mean_iou for each of [B, C, G, G] logits against an [H, W] annotation."""
    # Upsampled logits take B x C x H x W floats: a few scenes at a time.
    predicted = torch.cat([F.interpolate(chunk.float(), size=truth.shape, mode="bilinear", align_corners=False).argmax(1).cpu()
                           for chunk in logits.split(4)])
    labeled = truth != IGNORE_LABEL
    t = truth[labeled].cpu()
    scores = []
    for p in predicted[:, labeled.cpu()]:
        confusion = torch.bincount(t * num_classes + p, minlength=num_classes ** 2).reshape(num_classes, num_classes)
        inter = confusion.diag().double()
        union = confusion.sum(0) + confusion.sum(1) - confusion.diag()
        present = union > 0
        scores.append(float((inter[present] / union[present]).mean()))
    return scores


@torch.inference_mode()
def main(cfg: Config) -> None:
    scenes = list(cfg.scenes) + (random.Random(cfg.sample_seed).sample(validation_ids(), cfg.sample) if cfg.sample else [])
    assert scenes, "pass --scenes or --sample"
    device = torch.device(cfg.device)
    model = load_released_segmenter(scene_size_px=SCENE_PX, canvas_grid_size=CANVAS_GRID_SIZE, device=device).model
    per_seed = [seed_viewpoints(s, cfg.glimpses) for s in range(cfg.seeds)]
    viewpoints = [Viewpoint(centers=torch.cat([v[t].centers for v in per_seed]).to(device),
                            scales=torch.cat([v[t].scales for v in per_seed]).to(device)) for t in range(cfg.glimpses)]
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    checked = False
    with cfg.out.open("w") as out:
        for i, image_id in enumerate(scenes):
            image, annotation = load(image_id)
            truth = torch.from_numpy(annotation).long()
            scene = image.unsqueeze(0).to(device).expand(cfg.seeds, -1, -1, -1)
            state = model.init_state(batch_size=cfg.seeds, canvas_grid_size=CANVAS_GRID_SIZE)
            curves = []
            for t, viewpoint in enumerate(viewpoints):
                logits, state = model(glimpse=sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=GLIMPSE_SIZE_PX),
                                      state=state, viewpoint=viewpoint)
                scores = batched_miou(logits, truth, logits.shape[1])
                if not checked:
                    reference = mean_iou(logits[0].float().cpu().numpy(), annotation.astype(np.int64))
                    assert abs(reference - scores[0]) < 1e-6, f"batched mIoU {scores[0]} against mean_iou {reference}"
                    checked = True
                curves.append(scores)
            by_seed = np.array(curves).T  # [seeds, glimpses]
            monotone = [int(s) for s in np.flatnonzero((np.diff(by_seed, axis=1) >= 0).all(axis=1))]
            out.write(json.dumps({"image_id": image_id, "miou": by_seed.round(5).tolist(), "monotone_seeds": monotone}) + "\n")
            out.flush()
            log.info("%d/%d %s: %d monotone seeds of %d; best final mIoU %.3f", i + 1, len(scenes), image_id,
                     len(monotone), cfg.seeds, by_seed[:, -1].max())


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
