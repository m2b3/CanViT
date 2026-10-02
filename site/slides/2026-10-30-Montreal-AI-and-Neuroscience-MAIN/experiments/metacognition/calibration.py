"""Does the uncertainty read out from the canvas predict where the segmentation is wrong? Over ADE20K validation, with
CanViT-B and its ADE20K probe on the 64 x 64 canvas (the talk's model): per canvas cell, the entropy of the decoded
class distribution and whether the decoded class is right (the annotation at the cell's center pixel, unlabeled cells
left out), after the full-scene glimpse and after C2F's first five glimpses (the scene, then its quadrants).

Writes, per glimpse count: the AUROC of entropy for telling wrong cells from right ones (the probability that a wrong
cell has the higher entropy), the mean entropy of right and wrong cells in bits, and the accuracy in each entropy
decile."""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.benchmarks.ade20k import IGNORE_LABEL, dataset_root, decode_annotation
from canvit_pytorch.policies import make_policy
from canvit_pytorch.policies.entropy import predictive_entropy
from canvit_pytorch.preprocess import preprocess, preprocess_labels
from canvit_pytorch.viz.record import record
from canvit_pytorch.viz.released_model import CANVAS_GRID_SIZE, GLIMPSE_SIZE_PX, SCENE_SIZE_PX, load_released_segmenter
from canvit_pytorch.viz.web import cell_center_labels
from PIL import Image

from experiments import logs
from experiments.outputs import DECK_DATA

log = logging.getLogger(__name__)

GLIMPSE_COUNTS = (1, 5)


@dataclass(frozen=True)
class Config:
    out: Path = DECK_DATA / "metacognition/calibration.json"
    limit: int = 0
    """only the first N validation images; 0 for all"""
    device: str = "mps"


def auroc(scores_wrong: np.ndarray, scores_right: np.ndarray) -> float:
    """P(a wrong cell's score > a right cell's), ties counted half: the Mann-Whitney U statistic over its maximum."""
    scores = np.concatenate([scores_wrong, scores_right])
    ranks = np.empty(len(scores))
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    # Average ranks over ties.
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end + 1) / 2
    n_wrong, n_right = len(scores_wrong), len(scores_right)
    return float((ranks[:n_wrong].sum() - n_wrong * (n_wrong + 1) / 2) / (n_wrong * n_right))


def main(cfg: Config) -> None:
    root = dataset_root()
    images = sorted((root / "images/validation").glob("*.jpg"))
    if cfg.limit:
        images = images[:cfg.limit]
    device = torch.device(cfg.device)
    model = load_released_segmenter(scene_size_px=SCENE_SIZE_PX, canvas_grid_size=CANVAS_GRID_SIZE, device=device).model
    entropy = {n: [] for n in GLIMPSE_COUNTS}
    correct = {n: [] for n in GLIMPSE_COUNTS}
    torch.manual_seed(0)
    for i, path in enumerate(images):
        image = preprocess(SCENE_SIZE_PX)(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        pixels = preprocess_labels(SCENE_SIZE_PX)(Image.open(root / "annotations/validation" / f"{path.stem}.png"))
        truth = cell_center_labels(decode_annotation(pixels[0]).numpy(), CANVAS_GRID_SIZE).astype(np.int64)
        labeled = truth != IGNORE_LABEL
        policy = make_policy("coarse_to_fine", batch_size=1, device=device, num_glimpses=max(GLIMPSE_COUNTS),
                             canvas_grid_size=CANVAS_GRID_SIZE, canvas_logits=model.logits)
        rollout = record(model, image, policy, num_glimpses=max(GLIMPSE_COUNTS), canvas_grid_size=CANVAS_GRID_SIZE,
                         glimpse_size_px=GLIMPSE_SIZE_PX, capture_writes=False, annotation=None)
        for n in GLIMPSE_COUNTS:
            logits = torch.from_numpy(rollout.glimpses[n - 1].logits.astype(np.float64))
            cell_entropy = predictive_entropy(logits[None])[0].numpy() / math.log(2)  # bits
            entropy[n].append(cell_entropy[labeled])
            correct[n].append((logits.argmax(0).numpy() == truth)[labeled])
        if (i + 1) % 100 == 0:
            log.info("%d / %d images", i + 1, len(images))

    result = {"_about": __doc__.strip(), "images": len(images), "canvas_grid": CANVAS_GRID_SIZE, "by_glimpses": {}}
    for n in GLIMPSE_COUNTS:
        e, c = np.concatenate(entropy[n]), np.concatenate(correct[n])
        edges = np.quantile(e, np.linspace(0, 1, 11))
        decile = np.clip(np.searchsorted(edges, e, side="right") - 1, 0, 9)
        result["by_glimpses"][str(n)] = {
            "cells": int(len(e)), "accuracy": float(c.mean()),
            "auroc_entropy_for_errors": auroc(e[~c], e[c]),
            "mean_entropy_bits": {"right": float(e[c].mean()), "wrong": float(e[~c].mean())},
            "accuracy_by_entropy_decile": [float(c[decile == d].mean()) for d in range(10)],
            "entropy_decile_edges_bits": [float(v) for v in edges],
        }
        log.info("%d glimpses: %s", n, json.dumps({k: v for k, v in result["by_glimpses"][str(n)].items() if "decile" not in k}))
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(result, indent=1) + "\n")
    log.info("wrote %s", cfg.out)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
