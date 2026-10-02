"""Decode exported DINOv3 features with the released DINOv3 ViT-B/16 ADE20K linear probe (trained at 512 px), for the
step of the "DINOv3 feature maps" slide where a linear layer turns features into what's where. Prints, per exported
resolution, the probe's patch accuracy against the annotation and each class's IoU (patches), and writes
<exports>/<id>-<short side>-probs.npz (class probabilities per patch, float16)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES
from canvit_pytorch.hub.repos import released_dinov3_ade20k_probe
from canvit_pytorch.probes import SegmentationProbe

from experiments.outputs import WORK

PROBE_INPUT_PX = 512


@dataclass(frozen=True)
class Config:
    image_id: str
    short_sides: tuple[int, ...] = (512, 768, 1024)
    exports: Path = WORK / "foundation/exports"


def main(cfg: Config) -> None:
    probe = SegmentationProbe.from_pretrained(released_dinov3_ade20k_probe("dv3b", input_size_px=PROBE_INPUT_PX)).eval()
    assert not probe.use_ln, "DINOv3 features are already layer-normalized"
    for side in cfg.short_sides:
        data = np.load(cfg.exports / f"{cfg.image_id}-{side}.npz")
        grid_h, grid_w = (int(v) for v in data["grid"])
        features = torch.from_numpy(data["features"].astype(np.float32)).reshape(1, grid_h, grid_w, -1)
        with torch.inference_mode():
            probs = torch.softmax(probe(features), dim=1)[0].numpy()  # [C, H, W]
        predicted = probs.argmax(axis=0)
        truth = data["patch_labels"]
        labeled = truth != 255
        print(f"{side}px ({grid_w}x{grid_h}): patch accuracy {(predicted[labeled] == truth[labeled]).mean():.3f}")
        for cls in np.unique(truth[labeled]):
            inter = ((predicted == cls) & (truth == cls)).sum()
            union = ((predicted == cls) | (truth == cls))[labeled].sum()
            print(f"  {CLASS_NAMES[cls]:14s} IoU {inter / union:.3f}  mean p on its patches {probs[cls][truth == cls].mean():.3f}"
                  f"  patches {(truth == cls).sum()}")
        np.savez_compressed(cfg.exports / f"{cfg.image_id}-{side}-probs.npz", probs=probs.astype(np.float16))


if __name__ == "__main__":
    main(tyro.cli(Config))
