"""Run the quickstart slide's code (the talk's quickstart.py, as shown) on the street of the recorded bundles and save,
uncolored, what each of its model calls returned: the class index per canvas cell (labels-<n>.png, 64 x 64) and the
viewpoints. plot.py draws them."""

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
from PIL import Image

from experiments import logs
from experiments.outputs import DECK_DATA, SITE

log = logging.getLogger(__name__)

CODE = Path(__file__).resolve().parents[2] / "quickstart.py"


@dataclass(frozen=True)
class Config:
    bundle: Path = SITE / "data/street-c2f"
    """a recorded bundle of ADE_val_00001780: its scene.png (512 px, already square, which preprocess(512) keeps) and
    its readout's class names"""
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
    assert '"street.jpg"' in code, f"{CODE} no longer reads street.jpg"
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(cfg.bundle / "scene.png", Path(tmp) / "street.jpg")  # PIL reads the PNG whatever its name
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
    names = json.loads((cfg.bundle / "manifest.json").read_text())["readout"]["class_names"]
    (cfg.out / "export.json").write_text(json.dumps({
        "code": str(CODE.relative_to(SITE.parent)), "scene": str((cfg.bundle / "scene.png").relative_to(SITE.parent)),
        "viewpoints_row_col_scale": [[round(v, 4) for v in c["viewpoint"]] for c in calls], "class_names": names}, indent=1))
    for n, call in enumerate(calls, start=1):
        counts = np.bincount(call["labels"].ravel(), minlength=len(names))
        log.info("call %d: viewpoint %s, classes %s", n, call["viewpoint"],
                 ", ".join(f"{names[k]} {counts[k]}" for k in np.argsort(-counts)[:10] if counts[k]))


if __name__ == "__main__":
    logs.setup()
    main(tyro.cli(Config))
