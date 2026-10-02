"""Qualify and rank the three-object sequences of the sweep; print the distributions that put an example in context.
Writes the thresholds and the top candidates with their export ids (IMAGE_ID:CLASS#COMPONENT,...).

Per sequence, for object i (glimpsed at t = i, of three), with p the mean p(class) over the object's pixels as the
sweep scored it:
  seen_i    p after its own glimpse, the canvas kept
  kept_i    p after the third glimpse, the canvas kept
  reset_i   p after the third glimpse, the canvas reset before each glimpse (high for the last object by design)
  before_i  the highest p before its own glimpse, kept (the object lit up before it was looked at)
  shape_i   the fraction of its pixels lit (p >= definition.LIT_P) after the third glimpse, kept
  elsewhere the largest area lit away from its object, as a fraction of the scene, of any of the three classes: kept
            after any glimpse, or reset after the third (another instance of the class, or a guess)
A sequence qualifies when every seen_i >= MIN_SEEN, every kept_i >= MIN_KEPT, and reset_i <= MAX_RESET for the first
two objects. Candidates for the slide: sequences in one of the configured categories (PREFERRED by default) whose
photo's short side is at least MIN_SHORT_SIDE_PX, all three objects NAMEABLE, and with an object of a must_include
class when given; qualifying or not (too few qualify to fill a contact sheet); ranked by
  score = min(min_i seen_i, min_i kept_i) - max_{i<2} reset_i - ELSEWHERE_WEIGHT * elsewhere
          - BEFORE_WEIGHT * max_i before_i + SALIENT_BONUS * (number of SALIENT objects)
          - BORDER_PENALTY * (number of objects touching the scene's border: seen in part),
one sequence per image.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from experiments.memory.definition import slug
from experiments.outputs import WORK

MIN_SEEN, MIN_KEPT, MAX_RESET = 0.8, 0.7, 0.05
PREFERRED = ("street", "kitchen", "living_room", "dining_room", "bedroom")
MIN_SHORT_SIDE_PX = 480
NAMEABLE = {  # classes a general audience names at a glance
    "person", "car", "bed", "painting", "lamp", "sofa", "armchair", "chair", "swivel chair", "television", "clock",
    "toilet", "bicycle", "bus", "truck", "van", "boat", "ship", "airplane", "sink", "bathtub", "refrigerator", "oven",
    "microwave", "stove", "fireplace", "mirror", "plant", "flower", "vase", "bottle", "pillow", "cushion", "towel",
    "chandelier", "computer", "monitor", "table", "desk", "coffee table", "pool table", "bookcase", "wardrobe",
    "washer", "fan", "basket", "traffic light", "streetlight", "flag", "sculpture", "fountain", "tent", "minibike",
    "bench", "animal", "book", "pot", "dishwasher", "kitchen island", "stool", "ottoman", "palm", "tank", "barrel",
    "ball", "food", "plate", "glass", "crt screen", "screen", "arcade machine", "cradle", "plaything", "bag", "ashcan",
    "chest of drawers", "signboard",
}
SALIENT = {  # the author's examples (person, car, dog, bicycle, sofa, lamp, painting, plant, bottle) and their kin
    "person", "car", "animal", "bicycle", "minibike", "bus", "truck", "van", "sofa", "armchair", "lamp", "chandelier",
    "painting", "plant", "flower", "vase", "bottle", "television", "bed", "clock",
}
ELSEWHERE_WEIGHT, BEFORE_WEIGHT, SALIENT_BONUS, BORDER_PENALTY = 2.0, 0.5, 0.05, 0.1
PERCENTILES = (10, 25, 50, 75, 90)


@dataclass(frozen=True)
class Config:
    sweep: Path = WORK / "memory/sweep.json"
    out: Path = WORK / "memory/ranking.json"
    top: int = 48
    categories: tuple[str, ...] = PREFERRED
    """scene categories of slide candidates (any when empty)"""
    must_include: tuple[str, ...] = ()
    """classes of which a slide candidate must have an object (none required when empty)"""


def derived(r: dict) -> dict:
    kept, reset, k = np.array(r["prob"]["kept"]), np.array(r["prob"]["reset"]), len(r["objects"])  # [t, i]
    elsewhere_kept, elsewhere_reset = np.array(r["lit_elsewhere"]["kept"]), np.array(r["lit_elsewhere"]["reset"])
    seen = np.array([kept[i, i] for i in range(k)])
    before = np.array([kept[:i, i].max() if i else 0.0 for i in range(k)])
    areas = np.array([o["area"] for o in r["objects"]])  # of the scene
    elsewhere = float(max((elsewhere_kept * areas).max(), (elsewhere_reset[-1] * areas).max()))
    num_salient = sum(o["name"] in SALIENT for o in r["objects"])
    num_on_border = sum(o["touches_border"] for o in r["objects"])
    score = (min(seen.min(), kept[-1].min()) - reset[-1, :-1].max() - ELSEWHERE_WEIGHT * elsewhere
             - BEFORE_WEIGHT * before.max() + SALIENT_BONUS * num_salient - BORDER_PENALTY * num_on_border)
    return {"seen": seen, "kept": kept[-1], "reset": reset[-1], "before": before, "elsewhere": elsewhere,
            "shape": np.array(r["lit_on"]["kept"][-1]), "score": float(score)}


def qualifies(d: dict) -> bool:
    return bool(d["seen"].min() >= MIN_SEEN and d["kept"].min() >= MIN_KEPT and d["reset"][:-1].max() <= MAX_RESET)


def candidate_for_slide(r: dict, cfg: Config) -> bool:
    return ((not cfg.categories or r["category"] in cfg.categories) and r["short_side_px"] >= MIN_SHORT_SIDE_PX
            and all(o["name"] in NAMEABLE for o in r["objects"])
            and (not cfg.must_include or any(o["name"] in cfg.must_include for o in r["objects"])))


def quantiles(values: np.ndarray) -> str:
    return "  ".join(f"p{q} {v:.3f}" for q, v in zip(PERCENTILES, np.percentile(values, PERCENTILES)))


def distributions(label: str, ds: list[dict]) -> list[str]:
    if not ds:
        return [f"{label}: none"]
    stack = lambda key: np.stack([d[key] for d in ds])  # noqa: E731  [n, 3]
    seen, kept, reset = stack("seen"), stack("kept"), stack("reset")
    lines = [f"{label}: n = {len(ds)} sequences"]
    for i in range(seen.shape[1]):
        lines += [f"  object glimpsed at t = {i}:",
                  f"    seen    mean {seen[:, i].mean():.3f}  {quantiles(seen[:, i])}",
                  f"    kept    mean {kept[:, i].mean():.3f}  {quantiles(kept[:, i])}",
                  f"    reset   mean {reset[:, i].mean():.3f}  {quantiles(reset[:, i])}"]
    lines += [f"  min over the three, kept: mean {kept.min(1).mean():.3f}  {quantiles(kept.min(1))}",
              f"  max over the first two, reset: mean {reset[:, :-1].max(1).mean():.3f}  {quantiles(reset[:, :-1].max(1))}",
              f"  shape, min over the three: {quantiles(stack('shape').min(1))}",
              f"  elsewhere (fraction of the scene): {quantiles(np.array([d['elsewhere'] for d in ds]))}"]
    return lines


def export_id(r: dict) -> str:
    return f"{r['image_id']}:" + ",".join(f"{o['name']}#{o['component']}" for o in r["objects"])


def main(cfg: Config) -> None:
    sweep = json.loads(cfg.sweep.read_text())
    rows = sweep["sequences"]
    ds = [derived(r) for r in rows]
    recognized = [d for d in ds if d["seen"].min() >= MIN_SEEN]
    qualifying = [(r, d) for r, d in zip(rows, ds) if qualifies(d)]
    for_slide = [(r, d) for r, d in zip(rows, ds) if candidate_for_slide(r, cfg)]
    out = [(f"sweep {cfg.sweep.name}: {sweep['num_images']} images, {sweep['num_triples_tried']} object triples tried, "
            f"{len(rows)} admissible sequences in {len({r['image_id'] for r in rows})} images; geometry {sweep['geometry']}"),
           (f"qualifying: seen >= {MIN_SEEN} for all three, kept >= {MIN_KEPT} for all three, reset <= {MAX_RESET} for "
            f"the first two")]
    for label, keep in [("all three seen >= 0.8", lambda d: d["seen"].min() >= 0.8),
                        ("  and all kept >= 0.7", lambda d: d["seen"].min() >= 0.8 and d["kept"].min() >= 0.7),
                        ("  and all kept >= 0.8", lambda d: d["seen"].min() >= 0.8 and d["kept"].min() >= 0.8),
                        ("qualifying (kept >= 0.7, first two reset <= 0.05)", qualifies),
                        ("qualifying, with first two reset <= 0.02", lambda d: qualifies(d) and d["reset"][:-1].max() <= 0.02)]:
        selected = [r for r, d in zip(rows, ds) if keep(d)]
        out.append(f"  {label}: {len(selected)} sequences in {len({r['image_id'] for r in selected})} images")
    slide_qualifying = [(r, d) for r, d in for_slide if qualifies(d)]
    for label, chosen in [("slide candidates", for_slide), ("  of which qualifying", slide_qualifying)]:
        out.append(f"{label} ({'/'.join(cfg.categories) or 'any category'}, short side >= {MIN_SHORT_SIDE_PX} px, "
                   f"three nameable classes{', one of ' + '/'.join(cfg.must_include) if cfg.must_include else ''}): "
                   f"{len(chosen)} sequences in {len({r['image_id'] for r, _ in chosen})} images; by category "
                   f"{dict(Counter(r['category'] for r, _ in chosen))}")
    out += distributions("all admissible sequences", ds)
    out += distributions(f"all three recognized when glimpsed (seen >= {MIN_SEEN})", recognized)
    out += distributions("qualifying", [d for _, d in qualifying])
    ranked = sorted(for_slide, key=lambda rd: -rd[1]["score"])
    top, images = [], set()
    for r, d in ranked:
        if r["image_id"] in images:
            continue
        images.add(r["image_id"])
        top.append({
            "id": export_id(r), "slug": slug(r["image_id"], [o["name"] for o in r["objects"]]),
            "image_id": r["image_id"], "category": r["category"], "short_side_px": r["short_side_px"],
            "num_salient": sum(o["name"] in SALIENT for o in r["objects"]), "qualifies": qualifies(d),
            "score": round(d["score"], 4), "elsewhere": round(d["elsewhere"], 4),
            "objects": [{"name": o["name"], "component": o["component"], "area": round(o["area"], 4),
                         "touches_border": o["touches_border"], "seen": round(float(d["seen"][i]), 4),
                         "kept": round(float(d["kept"][i]), 4), "reset": round(float(d["reset"][i]), 4),
                         "before": round(float(d["before"][i]), 4), "shape": round(float(d["shape"][i]), 4)}
                        for i, o in enumerate(r["objects"])],
        })
    top = top[:cfg.top]
    out.append(f"top {len(top)} slide candidates (* qualifies; one sequence per image; salient count, score, "
               f"elsewhere as % of the scene; per object seen/kept/reset, b(efore), s(hape), area %):")
    for n, e in enumerate(top, start=1):
        objs = "  ".join(f"{o['name']} {o['seen']:.2f}/{o['kept']:.2f}/{o['reset']:.2f} b{o['before']:.2f} "
                         f"s{o['shape']:.2f} {100 * o['area']:.1f}%{' border' if o['touches_border'] else ''}"
                         for o in e["objects"])
        out.append(f"  {n:2d}{'*' if e['qualifies'] else ' '} {e['id']:<58} {e['category']:<12} salient "
                   f"{e['num_salient']} score {e['score']:.3f} elsewhere {100 * e['elsewhere']:.1f}%  {objs}")
    print("\n".join(out))
    cfg.out.write_text(json.dumps({
        "thresholds": {"min_seen": MIN_SEEN, "min_kept": MIN_KEPT, "max_reset": MAX_RESET,
                       "categories": cfg.categories, "must_include": cfg.must_include,
                       "min_short_side_px": MIN_SHORT_SIDE_PX, "border_penalty": BORDER_PENALTY,
                       "elsewhere_weight": ELSEWHERE_WEIGHT, "before_weight": BEFORE_WEIGHT,
                       "salient_bonus": SALIENT_BONUS, "salient": sorted(SALIENT)},
        "top": top}, indent=1))


if __name__ == "__main__":
    main(tyro.cli(Config))
