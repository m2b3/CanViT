"""Which second glimpse, after the full scene, changes the quickstart's segmentation most visibly on the street of the
recorded bundles (ADE_val_00001780): every viewpoint on a grid of centers and scales, scored by the canvas cells whose
label becomes right (against the bundle's 64 x 64 annotation) minus those that become wrong. Writes the viewpoints,
best first, and prints the top of them."""

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from canvit_pytorch import CanViTForSemanticSegmentation, Viewpoint, sample_at_viewpoint
from canvit_pytorch.preprocess import preprocess
from PIL import Image

from experiments import logs
from experiments.outputs import SITE, WORK

log = logging.getLogger(__name__)

BATCH = 32


@dataclass(frozen=True)
class Config:
    bundle: Path = SITE / "data/street-c2f"
    out: Path = WORK / "quickstart/sweep.json"
    device: str = "mps"


@torch.inference_mode()
def main(cfg: Config) -> None:
    device = torch.device(cfg.device)
    manifest = json.loads((cfg.bundle / "manifest.json").read_text())
    names = manifest["readout"]["class_names"]
    model = CanViTForSemanticSegmentation.from_pretrained_with_probe(
        pretrained_repo=manifest["model"]["repo"], probe_repo=manifest["readout"]["repo"]).eval().to(device)
    scene = preprocess(512)(Image.open(cfg.bundle / "scene.png").convert("RGB")).unsqueeze(0).to(device)
    truth = np.array(Image.open(cfg.bundle / "truth.png"))
    log.info("truth values %s ...", np.unique(truth)[:12])
    labeled = truth != 255 if 255 in truth else truth != 0

    state = model.init_state(batch_size=1, canvas_grid_size=64)
    full = Viewpoint.full_scene(batch_size=1, device=device)
    logits, state = model(glimpse=sample_at_viewpoint(spatial=scene, viewpoint=full, glimpse_size_px=128), state=state,
                          viewpoint=full)
    first = logits.argmax(1)[0].cpu().numpy()
    log.info("cells right after the full scene: %d of %d", int(((first == truth) & labeled).sum()), int(labeled.sum()))

    candidates = [(r, c, s) for s in (0.15, 0.2, 0.25, 0.33, 0.5)
                  for r, c in itertools.product(np.arange(-1 + s, 1 - s + 1e-6, 0.05), repeat=2)]
    rows = []
    for i in range(0, len(candidates), BATCH):
        chunk = candidates[i:i + BATCH]
        n = len(chunk)
        vp = Viewpoint(centers=torch.tensor([[r, c] for r, c, _ in chunk], dtype=torch.float32, device=device),
                       scales=torch.tensor([s for *_, s in chunk], dtype=torch.float32, device=device))
        batch_state = type(state)(**{k: v.expand(n, *v.shape[1:]).clone() for k, v in vars(state).items()})
        logits, _ = model(glimpse=sample_at_viewpoint(spatial=scene.expand(n, -1, -1, -1), viewpoint=vp, glimpse_size_px=128),
                          state=batch_state, viewpoint=vp)
        for (r, c, s), labels in zip(chunk, logits.argmax(1).cpu().numpy()):
            fixed = (labels == truth) & (first != truth) & labeled
            broken = (labels != truth) & (first == truth) & labeled
            gained = {names[k]: int(((labels == k) & fixed).sum()) for k in np.unique(labels[fixed])}
            rows.append({"row": round(float(r), 3), "col": round(float(c), 3), "scale": s,
                         "net": int(fixed.sum() - broken.sum()), "fixed": int(fixed.sum()), "broken": int(broken.sum()),
                         "gained": dict(sorted(gained.items(), key=lambda kv: -kv[1])[:4])})
    rows.sort(key=lambda r: -r["net"])
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(rows, indent=1))
    for r in rows[:25]:
        print(r)


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
