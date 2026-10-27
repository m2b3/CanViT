"""Candidate examples for "Processing a glimpse with a Vision Transformer": a zoomed glimpse on an object of ADE20K
validation that its pixels barely set apart from its surroundings but DINOv3 ViT-B/16's patch tokens do, as the slide
shows them (projections on the glimpse's first three principal components, as colors).

Per object (a connected component of a non-stuff class, as in looking_closer.definition), the glimpse is centered on
its bounding box with a side ZOOM_SIDE times the box's larger side. The 8 x 8 patches split into object patches (at
least OBJECT_SHARE of the patch is the object) and surroundings (at most SURROUNDINGS_SHARE), mixed ones left out. Two
separabilities of object from surroundings, each the AUC of Fisher's linear discriminant over the patches:
pixel_auc on each patch's mean CIELAB color, token_auc on each patch's displayed color (PCs 1-3, clipped as on the
slide). token_fisher is the discriminant's ratio of between-class to within-class variance, how far apart and how
uniform the two groups look. Writes every candidate with its measures, also every SAVE_EVERY images."""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
import tyro
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES, IGNORE_LABEL
from canvit_pytorch.teacher import TEACHER_REPO, load_teacher
from canvit_pytorch.viewpoint import sample_at_viewpoint

from experiments import logs
from experiments.ade20k import SCENE_PX, STUFF_IDS, load, pixels, validation_ids
from experiments.foundation.pca_colors import CLIP_PERCENTILES, colors, components
from experiments.glimpses import GLIMPSE_PX, ViewpointTuple, viewpoints
from experiments.outputs import WORK

log = logging.getLogger(__name__)

MIN_AREA, MAX_AREA = 0.003, 0.06  # the object's connected component, as a fraction of the scene
ZOOM_SIDE = 1.7  # the glimpse's side over the object's larger bounding-box side
MIN_SCALE, MAX_SCALE = 0.2, 0.4  # a glimpse covers at most 16% of the scene, and is upsampled at most 1.25 times
COMPONENTS_PER_CLASS = 2  # the largest ones that fit
PATCH_PX = 16
OBJECT_SHARE, SURROUNDINGS_SHARE = 0.75, 0.05
MIN_OBJECT_PATCHES, MIN_SURROUNDING_PATCHES = 8, 16
BATCH = 64
SAVE_EVERY = 200  # images
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class Candidate:
    image_id: str
    cls: int
    name: str
    component: int  # label of the object in ndi.label(labels == cls, EIGHT_CONNECTED)
    area: float  # of the scene
    bbox: tuple[int, int, int, int]  # (top, left, bottom, right), scene px, bottom and right exclusive
    viewpoint: ViewpointTuple


@dataclass(frozen=True)
class Config:
    out: Path = WORK / "vit_glimpse/candidates.json"
    device: str = "mps"
    limit: int | None = None
    """only the first images, for a quick check"""


def zoom_on(bbox: tuple[int, int, int, int]) -> ViewpointTuple:
    top, left, bottom, right = bbox
    s = ZOOM_SIDE * max(bottom - top, right - left) / SCENE_PX  # a glimpse's side is s * SCENE_PX px
    row, col = (top + bottom) / SCENE_PX - 1, (left + right) / SCENE_PX - 1
    return float(np.clip(row, s - 1, 1 - s)), float(np.clip(col, s - 1, 1 - s)), float(s)


def objects(image_id: str, labels: np.ndarray) -> list[tuple[Candidate, np.ndarray]]:
    found = []
    for cls in np.unique(labels):
        if cls == IGNORE_LABEL or cls in STUFF_IDS:
            continue
        parts, count = ndi.label(labels == cls, structure=EIGHT_CONNECTED)
        sizes = np.bincount(parts.ravel(), minlength=count + 1)
        kept = 0
        for component in np.argsort(-sizes[1:]) + 1:
            if kept == COMPONENTS_PER_CLASS or sizes[component] / SCENE_PX**2 < MIN_AREA:
                break
            if sizes[component] / SCENE_PX**2 > MAX_AREA:
                continue
            rows, cols = np.nonzero(parts == component)
            bbox = (int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1)
            vp = zoom_on(bbox)
            if not MIN_SCALE <= vp[2] <= MAX_SCALE:
                continue
            kept += 1
            found.append((Candidate(image_id, int(cls), CLASS_NAMES[cls], int(component),
                                    float(sizes[component] / SCENE_PX**2), bbox, vp), parts == component))
    return found


def patch_mean(x: np.ndarray) -> np.ndarray:
    """[GLIMPSE_PX, GLIMPSE_PX, C] to each patch's mean, [patches, C]."""
    grid = GLIMPSE_PX // PATCH_PX
    return x.reshape(grid, PATCH_PX, grid, PATCH_PX, -1).mean(axis=(1, 3)).reshape(grid * grid, -1)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """[..., 3] uint8 sRGB to CIELAB (D65)."""
    c = rgb / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(a positive scores above a negative), ties counting half."""
    diff = positive[:, None] - negative[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def discriminant(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(AUC, Fisher ratio) of Fisher's linear discriminant between [n, d] groups a and b, oriented so a scores higher."""
    within = np.cov(a, rowvar=False) * (len(a) - 1) + np.cov(b, rowvar=False) * (len(b) - 1)
    within += 1e-6 * np.trace(within) / len(within) * np.eye(len(within)) + 1e-12 * np.eye(len(within))
    w = np.linalg.solve(within, a.mean(0) - b.mean(0))
    pa, pb = a @ w, b @ w
    fisher = (pa.mean() - pb.mean()) ** 2 / (pa.var() + pb.var() + 1e-12)
    return auc(pa, pb), float(fisher)


def measure(teacher, batch: list[tuple[Candidate, np.ndarray, torch.Tensor]], device: torch.device) -> list[dict]:
    # The object's mask is resampled with the image, so both are what the glimpse sees.
    spatial = torch.stack([torch.cat([image, torch.from_numpy(mask).float()[None]]) for _, mask, image in batch]).to(device)
    crops = sample_at_viewpoint(spatial=spatial, viewpoint=viewpoints([c.viewpoint for c, _, _ in batch], device),
                                glimpse_size_px=GLIMPSE_PX)
    with torch.inference_mode():
        tokens = teacher(crops[:, :3]).patches.float().cpu().numpy()
    rows = []
    for i, (candidate, _, _) in enumerate(batch):
        share = patch_mean(crops[i, 3].cpu().numpy()[..., None])[:, 0]
        inside, outside = share >= OBJECT_SHARE, share <= SURROUNDINGS_SHARE
        if inside.sum() < MIN_OBJECT_PATCHES or outside.sum() < MIN_SURROUNDING_PATCHES:
            continue
        lab = patch_mean(srgb_to_lab(pixels(crops[i, :3])))
        shown = colors(components(tokens[i], 3), *CLIP_PERCENTILES)
        pixel_auc, pixel_fisher = discriminant(lab[inside], lab[outside])
        token_auc, token_fisher = discriminant(shown[inside], shown[outside])
        rows.append(asdict(candidate) | {
            "object_patches": int(inside.sum()), "surrounding_patches": int(outside.sum()),
            "object_share": float(share.mean()),
            "delta_e": float(np.linalg.norm(lab[inside].mean(0) - lab[outside].mean(0))),
            "pixel_auc": pixel_auc, "pixel_fisher": pixel_fisher, "token_auc": token_auc, "token_fisher": token_fisher,
            "token_distance": float(np.linalg.norm(shown[inside].mean(0) - shown[outside].mean(0))),
        })
    return rows


def main(cfg: Config) -> None:
    start = time.time()
    ids = validation_ids()[: cfg.limit]
    device = torch.device(cfg.device)
    teacher = load_teacher(TEACHER_REPO, device)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    pending: list[tuple[Candidate, np.ndarray, torch.Tensor]] = []
    for n, image_id in enumerate(ids, start=1):
        image, labels = load(image_id)
        pending += [(candidate, mask, image) for candidate, mask in objects(image_id, labels)]
        while len(pending) >= BATCH or (n == len(ids) and pending):
            rows += measure(teacher, pending[:BATCH], device)
            pending = pending[BATCH:]
        if n % SAVE_EVERY == 0 or n == len(ids):
            cfg.out.write_text(json.dumps({"model": TEACHER_REPO, "candidates": rows}, indent=1))
            log.info("%d/%d images, %d candidates, %.0fs", n, len(ids), len(rows), time.time() - start)
    log.info("done: %d candidates in %.0fs -> %s", len(rows), time.time() - start, cfg.out)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
