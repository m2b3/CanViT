"""Aggregate the sweep: mean recall and p(class) of the small objects after F, FZ and FF, overall, by object size,
and by class."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from experiments.glimpses import GLIMPSE_PX
from experiments.looking_closer.definition import CONDITIONS
from experiments.outputs import WORK

AREA_BINS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02)  # of the scene; upper edge inclusive in the last bin


@dataclass(frozen=True)
class Config:
    examples: Path = WORK / "looking_closer/examples.json"


def line(label: str, rows: list[dict]) -> str:
    means = {c: (np.mean([r[f"recall_{c}"] for r in rows]), np.mean([r[f"prob_{c}"] for r in rows])) for c in CONDITIONS}
    gain = np.array([r["recall_fz"] - r["recall_f"] for r in rows])
    sem = gain.std(ddof=1) / np.sqrt(len(gain)) if len(gain) > 1 else float("nan")
    return (f"{label:<26} n={len(rows):5d}  recall F {means['f'][0]:.3f}  FZ {means['fz'][0]:.3f}  FF {means['ff'][0]:.3f}"
            f"  (FZ-F {gain.mean():+.3f} ± {sem:.3f} s.e.)  p(class) F {means['f'][1]:.3f}  FZ {means['fz'][1]:.3f}"
            f"  FF {means['ff'][1]:.3f}  fp F {np.median([r['fp_f'] for r in rows]):.2f}"
            f"  FZ {np.median([r['fp_fz'] for r in rows]):.2f} (median)")


def main(cfg: Config) -> None:
    rows = json.loads(cfg.examples.read_text())
    print(line("all", rows))
    print(line("not touching the border", [r for r in rows if not r["touches_border"]]))
    print(f"by object area (side of an equal-area square in the {GLIMPSE_PX} px full-scene glimpse):")
    side = lambda a: np.sqrt(a) * GLIMPSE_PX  # noqa: E731
    for i, (low, high) in enumerate(zip(AREA_BINS, AREA_BINS[1:])):
        last = i == len(AREA_BINS) - 2
        binned = [r for r in rows if low <= r["area"] and (r["area"] <= high if last else r["area"] < high)]
        print(line(f"  {100 * low:.2f}-{100 * high:.2f}% ({side(low):.1f}-{side(high):.1f} px)", binned))
    print("by class (n >= 20):")
    names = sorted({r["name"] for r in rows}, key=lambda n: -sum(r["name"] == n for r in rows))
    for name in names:
        of = [r for r in rows if r["name"] == name]
        if len(of) >= 20:
            print(line(f"  {name}", of))
    print(f"{cfg.examples} has {len(rows)} objects")


if __name__ == "__main__":
    main(tyro.cli(Config))
