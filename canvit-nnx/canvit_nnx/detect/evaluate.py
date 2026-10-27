"""COCO average precision of a detector after each glimpse on val2017, under C2F viewpoints.

C2F draws a new order within each quadtree level for every scene and repeat,
so each repeat is an independent draw; the reported intervals are 95%
bootstrap intervals over repeats, and bound that viewpoint variation only.

    python -m canvit_nnx.detect.evaluate --detector <repo or directory> --data-dir <coco>
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from canvit_nnx.detect.data import (
    PADDING_IMAGE_ID,
    CocoImage,
    CocoSplit,
    ValidationBatch,
    annotation_file,
    normalize,
    read_split,
    validation_batches,
)
from canvit_nnx.detect.decode import Detections, decode
from canvit_nnx.detect.model import CanViTForObjectDetection, load_detector
from canvit_nnx.detect.scoring import bootstrap_interval, coco_detections, coco_statistics, load_ground_truth
from canvit_nnx.model.config import ComputeDtype
from canvit_nnx.policies import coarse_to_fine
from canvit_nnx.releases import RELEASED_CANVAS_GRID_SIZE, RELEASED_GLIMPSE_SIZE_PX, RELEASED_SCENE_SIZE_PX
from canvit_nnx.training.data_parallel import DataParallel
from canvit_nnx.viewpoint import Viewpoint, sample_at_viewpoint

log = logging.getLogger(__name__)

REPORTED_STATISTICS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


class Rollout(NamedTuple):
    """A scene's rollout, on the host."""

    scene: np.ndarray  # uint8 [S, S, 3]
    image: CocoImage
    detections: Detections[np.ndarray]  # [T, K] after each glimpse
    centers: np.ndarray  # [T, 2] of each glimpse's viewpoint
    scales: np.ndarray  # [T]


class Evaluation(NamedTuple):
    metrics: dict[str, float]  # on the leader host; empty on the others
    rollouts: list[Rollout]  # the first repeat's rollouts of the leader's first scenes; empty on the other hosts


@nnx.jit(static_argnames=("canvas_grid_size",))
def detections_after_each_glimpse(
    detector: CanViTForObjectDetection, scenes: jax.Array, viewpoints: tuple[Viewpoint, ...], *, canvas_grid_size: int
) -> Detections[jax.Array]:
    """Detections [T, B, K] after each glimpse of [B, S, S, 3] uint8 scenes."""
    normalized = normalize(scenes)
    state = detector.init_state(batch_size=scenes.shape[0], canvas_grid_size=canvas_grid_size)
    detections = []
    for viewpoint in viewpoints:
        glimpse = sample_at_viewpoint(spatial=normalized, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
        maps, state = detector(glimpse=glimpse, state=state, viewpoint=viewpoint)
        detections.append(decode(maps))
    return Detections(*(jnp.stack(field) for field in zip(*detections, strict=True)))


def average_precision(
    detector: CanViTForObjectDetection,
    split: CocoSplit,
    *,
    annotations: Path,
    data_parallel: DataParallel,
    batch_size: int,
    num_glimpses: int,
    repeats: int,
    key: jax.Array,
    canvas_grid_size: int,
    threads: int,
    num_rollouts: int = 0,
) -> Evaluation:
    """REPORTED_STATISTICS after each glimpse, mean over repeats, with their intervals when repeats > 1.

    Every host runs the rollouts; the leader scores them.
    """
    evaluated = nnx.view(detector, use_running_average=True)
    rows = data_parallel.host_rows(batch_size)
    batches = list(
        validation_batches(
            split.images, batch_size=batch_size, rows=rows, scene_size_px=RELEASED_SCENE_SIZE_PX, threads=threads
        )
    )
    image_of = {image.image_id: image for image in split.images}
    detection_rows: list[list[list[np.ndarray]]] = [[[] for _ in range(num_glimpses)] for _ in range(repeats)]
    rollouts: list[Rollout] = []
    for repeat in range(repeats):
        for index, batch in enumerate(batches):
            viewpoints = coarse_to_fine(
                jax.random.fold_in(jax.random.fold_in(key, repeat), index), batch_size=batch_size, num_glimpses=num_glimpses
            )
            scenes, image_ids = data_parallel.to_global(batch.scenes, batch.image_ids)
            detections = detections_after_each_glimpse(
                evaluated,
                scenes,
                data_parallel.split_full_batch(viewpoints, batch_size=batch_size),
                canvas_grid_size=canvas_grid_size,
            )
            boxes, scores, labels, image_ids = data_parallel.to_host(
                detections.boxes, detections.scores, detections.labels, image_ids
            )
            if not data_parallel.is_leader:
                continue
            real = image_ids != PADDING_IMAGE_ID
            images = [image_of[int(image_id)] for image_id in image_ids[real]]
            for glimpse in range(num_glimpses):
                detection_rows[repeat][glimpse].append(
                    coco_detections(
                        Detections(boxes[glimpse, real], scores[glimpse, real], labels[glimpse, real]),
                        image_ids=image_ids[real],
                        heights=np.array([image.height for image in images]),
                        widths=np.array([image.width for image in images]),
                        category_ids=split.category_ids,
                    )
                )
            if repeat == 0 and index == 0:
                host_detections = Detections(boxes, scores, labels)
                rollouts = [
                    _rollout(batch, local_row, image_of[int(image_id)], host_detections, image_ids, viewpoints, rows)
                    for local_row, image_id in enumerate(batch.image_ids[:num_rollouts])
                    if image_id != PADDING_IMAGE_ID
                ]
    if not data_parallel.is_leader:
        return Evaluation(metrics={}, rollouts=[])

    ground_truth = load_ground_truth(annotations)
    image_ids = np.array([image.image_id for image in split.images])
    statistics = [
        [
            coco_statistics(np.concatenate(rows_of_glimpse), ground_truth=ground_truth, image_ids=image_ids)
            for rows_of_glimpse in rows_of_repeat
        ]
        for rows_of_repeat in detection_rows
    ]
    return Evaluation(metrics=_summary(statistics, num_images=len(image_ids)), rollouts=rollouts)


def _rollout(
    batch: ValidationBatch,
    local_row: int,
    image: CocoImage,
    detections: Detections[np.ndarray],
    image_ids: np.ndarray,
    viewpoints: tuple[Viewpoint, ...],
    rows: slice,
) -> Rollout:
    """The rollout of this host's scene at local_row, found among the batch's gathered detections by image id."""
    row = int(np.flatnonzero(image_ids == image.image_id)[0])
    return Rollout(
        scene=batch.scenes[local_row],
        image=image,
        detections=Detections(*(field[:, row] for field in detections)),
        centers=np.stack([np.asarray(viewpoint.centers)[rows.start + local_row] for viewpoint in viewpoints]),
        scales=np.array([np.asarray(viewpoint.scales)[rows.start + local_row] for viewpoint in viewpoints]),
    )


def _summary(statistics: list[list[dict[str, float]]], *, num_images: int) -> dict[str, float]:
    """Means over repeats of statistics[repeat][glimpse], with 95% bootstrap intervals over more than one repeat."""
    repeats = len(statistics)
    metrics: dict[str, float] = {"val/images": num_images, "val/repeats": repeats}
    for glimpse in range(len(statistics[0])):
        for statistic in REPORTED_STATISTICS:
            values = [statistics[repeat][glimpse][statistic] for repeat in range(repeats)]
            name = f"val/{statistic}_t{glimpse}"
            metrics[name] = float(np.mean(values))
            if repeats > 1:
                metrics[f"{name}_ci_low"], metrics[f"{name}_ci_high"] = bootstrap_interval(values)
    return metrics


@dataclass(frozen=True)
class EvaluationConfig:
    """Score a CanViTForObjectDetection on COCO val2017; prints the metrics as JSON."""

    detector: str
    """A Hub repository or a directory holding the detector's config.json and model.safetensors."""
    data_dir: Path
    """COCO 2017: annotations/instances_val2017.json and val2017/."""
    num_glimpses: int = 21
    repeats: int = 5
    batch_size: int = 256
    """Scenes per batch, split over the devices."""
    canvas_grid_size: int = RELEASED_CANVAS_GRID_SIZE
    compute_dtype: ComputeDtype = "bfloat16"
    seed: int = 0
    """Seeds the viewpoints."""
    max_images: int | None = None
    """Score the first images only, by increasing id; None: all of val2017."""
    decode_threads: int = 32
    multi_host: bool = False
    """Run on every host of a multi-host slice, each launched with the same command."""


def evaluate(config: EvaluationConfig) -> dict[str, float]:
    data_parallel = DataParallel.start(multi_host=config.multi_host)
    split = read_split(config.data_dir, "val2017")
    if config.max_images is not None:
        split = split._replace(images=split.images[: config.max_images])
    with jax.set_mesh(data_parallel.mesh):
        detector = load_detector(config.detector, compute_dtype=config.compute_dtype)
        data_parallel.replicate(detector)
        return average_precision(
            detector,
            split,
            annotations=annotation_file(config.data_dir, "val2017"),
            data_parallel=data_parallel,
            batch_size=config.batch_size,
            num_glimpses=config.num_glimpses,
            repeats=config.repeats,
            key=jax.random.key(config.seed),
            canvas_grid_size=config.canvas_grid_size,
            threads=config.decode_threads,
        ).metrics


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
    metrics = evaluate(tyro.cli(EvaluationConfig, description=__doc__))
    if metrics:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

