"""Rank the scenes of the sweep as examples for the distillation slide, by the properties its OUTLINE.md entry asks for
("Example properties"), each scored by percentile among the candidates:

- unseen: the share of patches inside none of the eight glimpses;
- improvement: the gain from the first glimpse to the last in mean cosine similarity over never-seen patches and over
  all patches, the drop in PCA color distance over never-seen patches (what the audience sees), and how many of the
  seven later glimpses raise the never-seen similarity by at least GRADUAL_STEP (glimpse by glimpse, not at once);
- regions: how much of the teacher's PCA color variance the annotated classes explain, and how many classes have a
  distinct mean color (measures.class_separation).

Candidates: native short side at least the scene's 512 px (no upsampled scene), a scene category with at least
COMMON_CATEGORY scenes in ADE20K (sceneCategories.txt, training and validation; "misc" excluded), and at least
MIN_NEVER_SEEN of the patches never inside a glimpse. The score is the mean of the three property scores. Writes the
candidates, best first, and prints the top of them.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tyro

from experiments.ade20k import scene_categories
from experiments.distillation.sequences import GRID, NUM_GLIMPSES, SCENE_PX
from experiments.outputs import WORK

COMMON_CATEGORY = 100
MIN_NEVER_SEEN = 0.25
GRADUAL_STEP = 0.005
LAST = NUM_GLIMPSES - 1


@dataclass(frozen=True)
class Config:
    sweep: Path = WORK / "distillation/sweep.csv"
    out: Path = WORK / "distillation/ranked.csv"
    show: tuple[str, ...] = ("ADE_val_00001271",)
    """scenes whose rows are printed whatever their rank"""


def main(cfg: Config) -> None:
    table = pd.read_csv(cfg.sweep)
    counts = pd.Series(scene_categories()).value_counts()
    table["category_scenes"] = table.category.map(counts)
    table["never_seen_share"] = table.never_seen_patches / GRID**2
    table["gain_cos_never"] = table[f"cos_never_{LAST}"] - table.cos_never_0
    table["gain_cos_all"] = table[f"cos_all_{LAST}"] - table.cos_all_0
    table["gain_rgb_never"] = table.rgb_never_0 - table[f"rgb_never_{LAST}"]
    table["gain_zcos_never"] = table[f"zcos_never_{LAST}"] - table.zcos_never_0
    steps = table[[f"cos_never_{t}" for t in range(NUM_GLIMPSES)]].diff(axis=1).iloc[:, 1:]
    table["gradual_steps"] = (steps >= GRADUAL_STEP).sum(axis=1)
    table["worst_step_cos_all"] = table[[f"cos_all_{t}" for t in range(NUM_GLIMPSES)]].diff(axis=1).iloc[:, 1:].min(axis=1)

    native = table[["width", "height"]].min(axis=1) >= SCENE_PX
    common = (table.category_scenes >= COMMON_CATEGORY) & (table.category != "misc")
    unseen = table.never_seen_share >= MIN_NEVER_SEEN
    print(f"{len(table)} scenes; native short side >= {SCENE_PX} px: {native.sum()}; and a common category: "
          f"{(native & common).sum()}; and >= {MIN_NEVER_SEEN:.0%} of patches never seen: {(native & common & unseen).sum()}")
    candidates = table[native & common & unseen].copy()

    def pct(column: str) -> pd.Series:
        return candidates[column].rank(pct=True)

    candidates["score_unseen"] = pct("never_seen_share")
    candidates["score_improvement"] = (pct("gain_cos_never") + pct("gain_cos_all") + pct("gain_rgb_never")
                                       + pct("gradual_steps")) / 4
    candidates["score_regions"] = (pct("class_eta2") + pct("separated_classes")) / 2
    candidates["score"] = candidates[["score_unseen", "score_improvement", "score_regions"]].mean(axis=1)
    candidates = candidates.sort_values("score", ascending=False)
    candidates.to_csv(cfg.out, index=False)

    shown = ["image_id", "category", "score", "score_unseen", "score_improvement", "score_regions", "never_seen_share",
             "cos_all_0", f"cos_all_{LAST}", "cos_never_0", f"cos_never_{LAST}", "gain_rgb_never", "gain_zcos_never",
             "gradual_steps", "worst_step_cos_all", "class_eta2", "classes", "separated_classes", "teacher_pca3"]
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print(candidates[shown].head(40).to_string(index=False))
        print("\nCandidates, quantiles:")
        print(candidates[["never_seen_share", "gain_cos_never", "gain_cos_all", "gain_rgb_never", "gradual_steps",
                          "class_eta2", "separated_classes", f"cos_all_{LAST}"]].quantile([.1, .5, .9]).to_string())
        for image_id in cfg.show:
            row = table[table.image_id == image_id]
            print(f"\n{image_id} (native {int(row.width.iloc[0])} x {int(row.height.iloc[0])}, sequence {row.sequence.iloc[0]}):")
            print(row[[c for c in shown if c in row and not c.startswith("score")]].to_string(index=False))


if __name__ == "__main__":
    main(tyro.cli(Config))
