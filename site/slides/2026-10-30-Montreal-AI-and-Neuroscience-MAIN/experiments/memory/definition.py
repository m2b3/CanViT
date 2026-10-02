"""Three objects of distinct classes in an ADE20K validation scene, a glimpse on each, and CanViT's canvas decoded after
every glimpse, with the canvas kept from glimpse to glimpse and reset before each.

A sequence: objects O_0, O_1, O_2 of distinct classes (connected components of "thing" classes within Geometry's area
and scale bounds; per class, its top_per_class largest), ordered left to right by the column of their bounding-box
centers. Glimpse t is centered on O_t's bounding box, its side FILL_SIDE times the box's longer side (a square box
covers about half the glimpse), shifted to stay inside the scene; each glimpse has its own scale. A sequence is
admissible when no glimpse holds more than max_foreign of the pixels of another of the three objects, and no two
glimpses share more than max_glimpse_overlap of the smaller one's area.

Conditions, decoded after every glimpse t:
  kept   the canvas (the whole recurrent state) carried from glimpse to glimpse;
  reset  glimpse t alone, from a fresh state (t = 0 is the same as kept).

Per object i, condition and glimpse t (SCORES), with p = p(class of O_i):
  prob           the mean p over O_i's pixels;
  lit_on         the fraction of O_i's pixels with p >= LIT_P (how much of its shape shows);
  lit_elsewhere  the pixels farther than SPILL_MARGIN_PX from O_i with p >= LIT_P, over O_i's area (lit area
                 elsewhere, in object areas: other instances of the class, or a guess).
p comes from the probe's logits upsampled bilinearly to the scene, then softmax, as the paper's evaluation upsamples
before its argmax.
"""

import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES, IGNORE_LABEL, dataset_root
from canvit_pytorch.episode import run_episode
from canvit_pytorch.policies import MIN_SCALE, FixedSequence
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE
from PIL import Image

from experiments.ade20k import SCENE_PX, STUFF_IDS
from experiments.glimpses import GLIMPSE_PX, ViewpointTuple, viewpoints

NUM_OBJECTS = 3
FILL_SIDE = 1.4  # glimpse side over the object's larger bounding-box side
SPILL_MARGIN_PX = 8  # one canvas patch at a 64 x 64 canvas: bilinear upsampling blurs the object's edge this far
LIT_P = 0.5  # p from which a pixel counts as lit
SCORES = ("prob", "lit_on", "lit_elsewhere")
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)

Condition = Literal["kept", "reset"]
CONDITIONS: tuple[Condition, ...] = ("kept", "reset")


@dataclass(frozen=True)
class SceneObject:
    image_id: str
    cls: int
    name: str
    component: int  # label of the object in ndi.label(labels == cls, EIGHT_CONNECTED)
    area: float  # of the scene
    class_area: float  # of the scene, every pixel labeled cls
    num_components: int  # of cls in the scene
    bbox: tuple[int, int, int, int]  # (top, left, bottom, right), scene px, bottom and right exclusive
    touches_border: bool


@dataclass(frozen=True)
class Geometry:
    min_area: float = 0.005
    """of the scene, an object's connected component"""
    max_area: float = 0.15
    max_scale: float = 0.5
    """of an object's glimpse: its side over the scene's; 0.5 covers a quarter of the scene"""
    top_per_class: int = 2
    """largest components of a class tried as its object"""
    max_foreign: float = 0.10
    """of an object's pixels inside another object's glimpse"""
    max_glimpse_overlap: float = 0.30
    """of the smaller glimpse's area, between any two glimpses"""


@dataclass(frozen=True)
class Candidate:
    objs: tuple[SceneObject, ...]  # in glimpse order
    masks: tuple[np.ndarray, ...]  # [S, S] bool each
    viewpoints: tuple[ViewpointTuple, ...]
    foreign: tuple[tuple[float, ...], ...]  # foreign[j][i]: fraction of O_i's pixels inside glimpse j
    max_glimpse_overlap: float


def short_side_px(image_id: str) -> int:
    with Image.open(dataset_root() / "images/validation" / f"{image_id}.jpg") as image:
        return min(image.size)


def required_scale(bbox: tuple[int, int, int, int]) -> float:
    top, left, bottom, right = bbox
    return max(FILL_SIDE * max(bottom - top, right - left) / SCENE_PX, MIN_SCALE)  # a crop's side is scale * S px


def objects(image_id: str, labels: np.ndarray, geometry: Geometry) -> list[tuple[SceneObject, np.ndarray]]:
    """Every connected component of a non-stuff class within the geometry's area bounds that a glimpse of at most its
    max_scale frames, with its [S, S] mask."""
    found = []
    for cls in np.unique(labels):
        if cls == IGNORE_LABEL or cls in STUFF_IDS:
            continue
        mask = labels == cls
        components, count = ndi.label(mask, structure=EIGHT_CONNECTED)
        sizes = np.bincount(components.ravel(), minlength=count + 1)
        for component in range(1, count + 1):
            if not geometry.min_area <= sizes[component] / SCENE_PX**2 <= geometry.max_area:
                continue
            component_mask = components == component
            rows, cols = np.nonzero(component_mask)
            bbox = (int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1)
            if required_scale(bbox) > geometry.max_scale:
                continue
            obj = SceneObject(
                image_id=image_id, cls=int(cls), name=CLASS_NAMES[cls], component=component,
                area=float(sizes[component] / SCENE_PX**2), class_area=float(mask.mean()), num_components=int(count),
                bbox=bbox, touches_border=bbox[0] == 0 or bbox[1] == 0 or bbox[2] == SCENE_PX or bbox[3] == SCENE_PX,
            )
            found.append((obj, component_mask))
    return found


def covering_box_px(vp: ViewpointTuple, size: int = SCENE_PX) -> tuple[int, int, int, int]:
    """(top, left, bottom, right) of a viewpoint's crop in a size-px scene, rounded outward."""
    row, col, s = vp
    return (int(np.floor((row - s + 1) * size / 2)), int(np.floor((col - s + 1) * size / 2)),
            int(np.ceil((row + s + 1) * size / 2)), int(np.ceil((col + s + 1) * size / 2)))


def clipped_box(vp: ViewpointTuple) -> tuple[int, int, int, int]:
    top, left, bottom, right = covering_box_px(vp)
    return max(top, 0), max(left, 0), min(bottom, SCENE_PX), min(right, SCENE_PX)


def fraction_inside(mask: np.ndarray, vp: ViewpointTuple) -> float:
    top, left, bottom, right = clipped_box(vp)
    return float(mask[top:bottom, left:right].sum() / mask.sum())


def overlap_of_smaller(a: ViewpointTuple, b: ViewpointTuple) -> float:
    (at, al, ab, ar), (bt, bl, bb, br) = clipped_box(a), clipped_box(b)
    shared = max(0, min(ab, bb) - max(at, bt)) * max(0, min(ar, br) - max(al, bl))
    return shared / min((ab - at) * (ar - al), (bb - bt) * (br - bl))


def glimpse_on(obj: SceneObject) -> ViewpointTuple:
    s = required_scale(obj.bbox)
    top, left, bottom, right = obj.bbox
    row, col = (top + bottom) / SCENE_PX - 1, (left + right) / SCENE_PX - 1
    vp = float(np.clip(row, s - 1, 1 - s)), float(np.clip(col, s - 1, 1 - s)), s
    box = covering_box_px(vp)
    assert box[0] <= top and box[1] <= left and box[2] >= bottom and box[3] >= right, (vp, obj)
    return vp


def left_to_right(objs: list[tuple[SceneObject, np.ndarray]]) -> list[tuple[SceneObject, np.ndarray]]:
    center = lambda o: ((o.bbox[1] + o.bbox[3]) / 2, (o.bbox[0] + o.bbox[2]) / 2)  # noqa: E731  (col, row)
    return sorted(objs, key=lambda om: center(om[0]))


def candidate(objs: list[tuple[SceneObject, np.ndarray]], geometry: Geometry) -> Candidate | None:
    """The sequence of the module docstring for these objects, or None when it is not admissible."""
    assert len({o.cls for o, _ in objs}) == len(objs), [o.name for o, _ in objs]
    objs = left_to_right(objs)
    vps = [glimpse_on(o) for o, _ in objs]
    foreign = [[fraction_inside(mask, vp) for _, mask in objs] for vp in vps]
    if any(foreign[j][i] > geometry.max_foreign for j in range(len(objs)) for i in range(len(objs)) if i != j):
        return None
    overlap = max(overlap_of_smaller(a, b) for a, b in itertools.combinations(vps, 2))
    if overlap > geometry.max_glimpse_overlap:
        return None
    return Candidate(objs=tuple(o for o, _ in objs), masks=tuple(m for _, m in objs), viewpoints=tuple(vps),
                     foreign=tuple(tuple(round(f, 4) for f in row) for row in foreign), max_glimpse_overlap=overlap)


def candidates(image_id: str, labels: np.ndarray, geometry: Geometry) -> tuple[list[Candidate], int]:
    """Every admissible sequence of the scene, and how many object triples were tried."""
    by_class: dict[int, list[tuple[SceneObject, np.ndarray]]] = {}
    for o, mask in objects(image_id, labels, geometry):
        by_class.setdefault(o.cls, []).append((o, mask))
    pools = [sorted(of_class, key=lambda om: -om[0].area)[:geometry.top_per_class] for of_class in by_class.values()]
    tried, admissible = 0, []
    for classes in itertools.combinations(pools, NUM_OBJECTS):
        for objs in itertools.product(*classes):
            tried += 1
            if (c := candidate(list(objs), geometry)) is not None:
                admissible.append(c)
    return admissible, tried


@torch.inference_mode()
def rollouts(model, images: torch.Tensor, sequences: list[tuple[ViewpointTuple, ...]]) -> dict[Condition, list[torch.Tensor]]:
    """images [B, 3, S, S]; sequences[b][t]: example b's viewpoint at glimpse t -> per condition, [B, C, G, G]
    float32 logits after every glimpse."""
    batch, steps, device = len(sequences), len(sequences[0]), images.device
    assert images.shape[0] == batch and all(len(seq) == steps for seq in sequences)
    fresh = lambda n: model.canvit.init_state(batch_size=n, canvas_grid_size=CANVAS_GRID_SIZE)  # noqa: E731
    episode = dict(canvit=model.canvit, glimpse_size_px=GLIMPSE_PX)
    kept = run_episode(**episode, images=images, num_glimpses=steps, initial_state=fresh(batch),
                       policy=FixedSequence([viewpoints([seq[t] for seq in sequences], device) for t in range(steps)]))
    # Glimpses 1.. each from a fresh state, as one batch ordered glimpse-major.
    later = [seq[t] for t in range(1, steps) for seq in sequences]
    (alone,) = run_episode(**episode, images=images.repeat(steps - 1, 1, 1, 1), num_glimpses=1,
                           initial_state=fresh(batch * (steps - 1)), policy=FixedSequence([viewpoints(later, device)]))
    kept_logits = [model.logits(step.state.canvas).float() for step in kept]
    return {"kept": kept_logits, "reset": [kept_logits[0], *model.logits(alone.state.canvas).float().split(batch)]}


@dataclass(frozen=True)
class Target:
    """An object scored in its scene, on the device of the logits."""
    cls: int
    mask: torch.Tensor  # [S, S] bool
    far: torch.Tensor  # [S, S] bool: farther than SPILL_MARGIN_PX from the object


def targets(c: Candidate, device: torch.device) -> list[Target]:
    def far(mask: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(~ndi.binary_dilation(mask, iterations=SPILL_MARGIN_PX)).to(device)
    return [Target(o.cls, torch.from_numpy(m).to(device), far(m)) for o, m in zip(c.objs, c.masks)]


def class_probabilities(logits: torch.Tensor, classes: list[int]) -> torch.Tensor:
    """[B, C, g, g] logits -> [B, len(classes), S, S] p, upsampled bilinearly then softmaxed over all C classes: the
    paper's evaluation, used for scoring."""
    up = F.interpolate(logits, size=(SCENE_PX, SCENE_PX), mode="bilinear", align_corners=False)
    return up.softmax(1)[:, classes]


def cell_probabilities(logits: torch.Tensor, classes: list[int]) -> torch.Tensor:
    """[B, C, g, g] logits -> [B, len(classes), g, g] p per canvas cell: what figures show, upsampled nearest only
    [Yohaï, 2026-10-01: "it should be hard nearest for dinov3 stuff, probability maps etc!"]."""
    return logits.softmax(1)[:, classes]


def nearest_to_scene(cells: np.ndarray) -> np.ndarray:
    """[..., g, g] -> [..., S, S], each canvas cell repeated over its S / g pixels."""
    repeat = SCENE_PX // cells.shape[-1]
    assert repeat * cells.shape[-1] == SCENE_PX, cells.shape
    return cells.repeat(repeat, axis=-2).repeat(repeat, axis=-1)


@torch.inference_mode()
def scores(logits: torch.Tensor, scenes: list[list[Target]]) -> np.ndarray:
    """logits [B, C, g, g] after one glimpse; scenes[b]: the objects of sequence b -> [B, k, len(SCORES)]."""
    rows = []
    for b, scene in enumerate(scenes):  # one at a time: [1, 150, 512, 512] float32 is 157 MB
        p = class_probabilities(logits[b:b + 1], [t.cls for t in scene])[0]
        for t, p_class in zip(scene, p):
            lit, area = p_class >= LIT_P, t.mask.sum()
            rows.append(torch.stack([p_class[t.mask].mean(), lit[t.mask].sum() / area, lit[t.far].sum() / area]))
    k = len(scenes[0])
    assert all(len(scene) == k for scene in scenes)
    return torch.stack(rows).cpu().double().numpy().reshape(len(scenes), k, len(SCORES))


def slug(image_id: str, names: list[str]) -> str:
    return "-".join([image_id, *(n.replace(" ", "_") for n in names)])
