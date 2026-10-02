"""Small objects of ADE20K validation, a zoomed viewpoint on each, and CanViT's logits after the full-scene glimpse
alone (F), after F then the zoom (FZ), and after F then F again (FF, the control: a second glimpse without zoom)."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES, IGNORE_LABEL
from canvit_pytorch.episode import run_episode
from canvit_pytorch.policies import MIN_SCALE, FixedSequence
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE

from experiments.ade20k import SCENE_PX, STUFF_IDS
from experiments.glimpses import FULL_SCENE, GLIMPSE_PX, ViewpointTuple, viewpoints

Condition = Literal["f", "fz", "ff"]
CONDITIONS: tuple[Condition, ...] = ("f", "fz", "ff")
MIN_AREA, MAX_AREA = 0.0005, 0.02  # the object's connected component, as a fraction of the scene
ZOOM_SIDE = 2.5  # the zoomed glimpse's side over the object's larger bounding-box side
MAX_ZOOM_SCALE = 0.4  # objects needing a wider glimpse (long, thin ones) are skipped: that would not be a zoom
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class SmallObject:
    image_id: str
    cls: int
    name: str
    component: int  # label of the object in ndi.label(labels == cls, EIGHT_CONNECTED)
    area: float  # of the scene
    class_area: float  # of the scene, every pixel labeled cls
    num_components: int  # of cls in the scene
    bbox: tuple[int, int, int, int]  # (top, left, bottom, right), scene px, bottom and right exclusive
    touches_border: bool
    zoom: ViewpointTuple


def zoom_on(bbox: tuple[int, int, int, int]) -> ViewpointTuple:
    """A viewpoint centered on bbox, its side ZOOM_SIDE times bbox's larger side, at least MIN_SCALE, inside the scene."""
    top, left, bottom, right = bbox
    s = max(ZOOM_SIDE * max(bottom - top, right - left) / SCENE_PX, MIN_SCALE)  # a crop's side is s * S px
    row, col = (top + bottom) / SCENE_PX - 1, (left + right) / SCENE_PX - 1
    return float(np.clip(row, s - 1, 1 - s)), float(np.clip(col, s - 1, 1 - s)), s


def small_objects(image_id: str, labels: np.ndarray) -> list[tuple[SmallObject, np.ndarray]]:
    """Per non-stuff class, its largest connected component of MIN_AREA..MAX_AREA of the scene that a zoom of scale
    at most MAX_ZOOM_SCALE covers, with its [S, S] mask."""
    found = []
    for cls in np.unique(labels):
        if cls == IGNORE_LABEL or cls in STUFF_IDS:
            continue
        mask = labels == cls
        components, count = ndi.label(mask, structure=EIGHT_CONNECTED)
        sizes = np.bincount(components.ravel(), minlength=count + 1)
        best = None
        for component in np.argsort(-sizes[1:]) + 1:
            if not MIN_AREA <= sizes[component] / SCENE_PX**2 <= MAX_AREA:
                continue
            rows, cols = np.nonzero(components == component)
            bbox = (int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1)
            zoom = zoom_on(bbox)
            if zoom[2] <= MAX_ZOOM_SCALE:
                best = (int(component), bbox, zoom)
                break
        if best is None:
            continue
        component, bbox, zoom = best
        obj = SmallObject(
            image_id=image_id, cls=int(cls), name=CLASS_NAMES[cls], component=component,
            area=float(sizes[component] / SCENE_PX**2), class_area=float(mask.mean()), num_components=int(count),
            bbox=bbox, touches_border=bbox[0] == 0 or bbox[1] == 0 or bbox[2] == SCENE_PX or bbox[3] == SCENE_PX,
            zoom=zoom,
        )
        found.append((obj, components == component))
    return found


@torch.inference_mode()
def canvit_logits(model, images: torch.Tensor, zooms: list[ViewpointTuple]) -> dict[Condition, torch.Tensor]:
    """[B, C, G, G] float32 logits decoded from the canvas after F, FZ and FF; FF continues from F's state."""
    batch, device = images.shape[0], images.device
    full = viewpoints([FULL_SCENE] * batch, device)
    episode = dict(canvit=model.canvit, images=images, glimpse_size_px=GLIMPSE_PX)
    f, fz = run_episode(**episode, policy=FixedSequence([full, viewpoints(zooms, device)]), num_glimpses=2,
                        initial_state=model.canvit.init_state(batch_size=batch, canvas_grid_size=CANVAS_GRID_SIZE))
    (ff,) = run_episode(**episode, policy=FixedSequence([full]), num_glimpses=1, initial_state=f.state)
    return {cond: model.logits(step.state.canvas).float() for cond, step in zip(CONDITIONS, (f, fz, ff))}


@torch.inference_mode()
def scene_maps(logits: torch.Tensor, classes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """[B, C, g, g] logits -> ([B, S, S] argmax labels, [B, S, S] p(classes[b])), from logits bilinearly upsampled to
    the scene, as the paper's evaluation does."""
    labels, probs = [], []
    for chunk, cls in zip(logits.split(4), torch.tensor(classes, device=logits.device).split(4)):
        up = F.interpolate(chunk, size=(SCENE_PX, SCENE_PX), mode="bilinear", align_corners=False)
        labels.append(up.argmax(1).cpu())
        probs.append(up.softmax(1)[torch.arange(len(cls), device=up.device), cls].cpu())
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()
