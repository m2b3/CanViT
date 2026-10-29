"""Run the quickstart slide's code (the talk's quickstart.py, as shown) on an ADE20K validation scene, chosen by
sweep.py and by eye, and save, uncolored, the scene (scene.png, 512 px) and what each of its model calls returned: the
class index per canvas cell (labels-<n>.png, 64 x 64) and the viewpoints. plot.py draws them."""

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from canvit_pytorch import CanViTForSemanticSegmentation
from canvit_pytorch.benchmarks.ade20k import CLASS_NAMES
from PIL import Image

from experiments import logs
from experiments.ade20k import load, pixels
from experiments.outputs import DECK_DATA, SITE

log = logging.getLogger(__name__)

CODE = Path(__file__).resolve().parents[2] / "quickstart.py"


@dataclass(frozen=True)
class Config:
    image_id: str = "ADE_val_00000996"
    """the scene: a bus on a seafront street (chosen by eye among sweep.py's candidates)"""
    out: Path = DECK_DATA / "quickstart"


def main(cfg: Config) -> None:
    calls = []
    forward = CanViTForSemanticSegmentation.forward

    def recorded(self, *, glimpse, state, viewpoint):
        logits, state = forward(self, glimpse=glimpse, state=state, viewpoint=viewpoint)
        calls.append({"labels": logits.argmax(1)[0].cpu().numpy().astype(np.uint8),
                      "viewpoint": [*viewpoint.centers[0].tolist(), float(viewpoint.scales[0])]})
        return logits, state

    CanViTForSemanticSegmentation.forward = recorded
    code = CODE.read_text()
    assert '"scene.jpg"' in code, f"{CODE} no longer reads scene.jpg"
    cfg.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels(load(cfg.image_id)[0])).save(cfg.out / "scene.png")  # 512 px, square: preprocess(512) keeps it
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(cfg.out / "scene.png", Path(tmp) / "scene.jpg")  # PIL reads the PNG whatever its name
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            exec(compile(code, str(CODE), "exec"), {"__name__": "__main__"})
        finally:
            os.chdir(cwd)
            CanViTForSemanticSegmentation.forward = forward
    assert len(calls) == code.count("= model("), (len(calls), code.count("= model("))
    cfg.out.mkdir(parents=True, exist_ok=True)
    for n, call in enumerate(calls, start=1):
        Image.fromarray(call["labels"]).save(cfg.out / f"labels-{n}.png")
    names = list(CLASS_NAMES)
    (cfg.out / "export.json").write_text(json.dumps({
        "code": str(CODE.relative_to(SITE.parent)), "scene": str((cfg.out / "scene.png").relative_to(SITE.parent)), "image_id": cfg.image_id,
        "viewpoints_row_col_scale": [[round(v, 4) for v in c["viewpoint"]] for c in calls], "class_names": names}, indent=1))
    for n, call in enumerate(calls, start=1):
        counts = np.bincount(call["labels"].ravel(), minlength=len(names))
        log.info("call %d: viewpoint %s, classes %s", n, call["viewpoint"],
                 ", ".join(f"{names[k]} {counts[k]}" for k in np.argsort(-counts)[:10] if counts[k]))


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
