"""ADE20K mIoU after each glimpse of the published learned viewing policies (canvit_pytorch.rl): both objectives, each
seed's scorer as its run kept it by validation and as its last training step left it, evaluated by
canvit_pytorch.evaluate with the observer and probe of the paper's canvas-64 policy curves, beside that table's C2F
and EG-C2F. Each result names the scorer it evaluated; the evaluation record (checkpoints.json) ties that scorer to its
training run, step and manifest, the source of the recipe."""

from collections import Counter

import numpy as np
import ultraplot as uplt
from matplotlib.figure import Figure as MplFigure
from matplotlib.ticker import NullLocator

from canvit_paper_exporter.ade20k import seg
from canvit_paper_exporter.core import Dataset, Figure, load_json
from canvit_paper_exporter.paths import Inputs
from canvit_paper_exporter.results import Result, Viewing, group_by, load_results, miou_curve, results_in
from canvit_paper_exporter.stats import CONFIDENCE, per_timestep_ci
from canvit_paper_exporter.style import (
    LABEL_SIZE, LEARNED_POLICY_STYLES, POLICY_STYLES, SINGLE_PANEL_FIGHEIGHT, SINGLE_PANEL_FIGWIDTH, TICK_SIZE,
    bottom_legend_with_headers, mean_and_ci,
)

ENTRY = "ade20k_seg_learned_policies"
RESULTS = results_in(ENTRY)
# The evaluation record: for each result, the scorer it evaluated, its training run, step and manifest; and the
# training time of each run.
RECORD = Inputs(ENTRY, ("checkpoints.json", "training_durations.json"))
OBJECTIVES = {"q": "q_regression", "pg": "policy_gradient"}
SELECTIONS = {"val-selected": "validation", "last step": "last"}
BASELINE_POLICIES = ("coarse_to_fine", "entropy_coarse_to_fine")
# The manifest keys of the recipe every run shares; the others name the run, its paths or its objective.
SHARED_RECIPE = (
    "scene_size", "canvas_grid", "glimpse_px", "model_repo", "scales", "centers_per_axis", "width", "block_layers",
    "feature_groups", "train_horizon", "score_res", "batch_size", "budget_forwards", "lr", "warmup_frac",
    "weight_decay", "adam_beta1", "adam_beta2", "grad_clip", "target_momentum", "eval_every",
)


def _single[T](values: list[T]) -> T:
    distinct = {repr(v): v for v in values}
    assert len(distinct) == 1, f"expected one value, got {list(distinct)}"
    return next(iter(distinct.values()))


def _recipe(manifests: list[dict]) -> dict:
    shared = _single([{k: m["config"][k] for k in SHARED_RECIPE} for m in manifests])
    num_glimpses = shared["train_horizon"] + 1
    prime_on_policy = _single([m["config"]["prime_on_policy"] for m in manifests if m["kind"] == "q"])
    pg_objective = _single([m["config"]["objective"] for m in manifests if m["kind"] == "pg"])
    assert pg_objective["credit"] == "immediate", pg_objective
    steps = shared["budget_forwards"] // (shared["batch_size"] * num_glimpses)
    return {
        "model_repo": shared["model_repo"], "scene_size": shared["scene_size"], "canvas_grid": shared["canvas_grid"],
        "glimpse_px": shared["glimpse_px"], "scales": shared["scales"], "centers_per_axis": shared["centers_per_axis"],
        "num_candidates": len(shared["scales"]) * shared["centers_per_axis"] ** 2, "num_glimpses": num_glimpses,
        "batch_size": shared["batch_size"], "glimpse_budget": shared["budget_forwards"],
        "steps": steps, "lr": shared["lr"], "weight_decay": shared["weight_decay"],
        "warmup_steps": int(shared["warmup_frac"] * steps), "validation_interval": shared["eval_every"],
        "adam_betas": [shared["adam_beta1"], shared["adam_beta2"]], "grad_clip": shared["grad_clip"],
        # The July code took the scorer's choice with probability prime_on_policy, a random candidate otherwise.
        "q_regression_epsilon": 1 - prime_on_policy, "policy_gradient_entropy_floor": pg_objective["entropy_target"],
    }


def _learned(results: list[Result], records: list[dict]) -> tuple[list[dict], dict]:
    rows: dict[tuple[str, str], list[tuple[int, int, list[float]]]] = {}
    manifests = []
    for result in results:
        [record] = [r for r in records if r["scorer"] == result["metadata"]["scorer"]]
        manifest = record["training_manifest"]
        manifests.append(manifest)
        key = (OBJECTIVES[manifest["kind"]], SELECTIONS[record["selection"]])
        rows.setdefault(key, []).append((record["seed"], record["training_step"], miou_curve(result)))
    curves = []
    for (objective, selection), entries in sorted(rows.items()):
        entries.sort()
        seeds = [seed for seed, _, _ in entries]
        assert seeds == sorted(set(seeds)), (objective, selection, seeds)
        curves.append({
            "objective": objective, "selection": selection, "seeds": seeds,
            "training_steps": [step for _, step, _ in entries],
            "per_timestep": per_timestep_ci(np.array([curve for _, _, curve in entries])),
        })
    return curves, _recipe(manifests)


def _baselines(recipe: dict) -> list[dict]:
    canvas = Viewing("", recipe["scene_size"], recipe["canvas_grid"])
    runs = [r for r in load_results(seg.RESULTS).values() if "variant" not in r["metadata"]["config"]]
    grouped = group_by((r for r in runs if r["metadata"]["config"]["episode"]["num_glimpses"] > 1), seg.flagship_viewing)
    curves = []
    for policy in BASELINE_POLICIES:
        policy_runs = grouped[canvas._replace(policy=policy)]
        curves.append({
            "policy": policy, "n_runs": len(policy_runs),
            "per_timestep": per_timestep_ci(np.array([miou_curve(r) for r in policy_runs])),
        })
    return curves


def _training_time() -> dict:
    """The range of the evaluated training runs' wall-clock minutes, validation included, and the device."""
    durations = load_json(RECORD.file("training_durations.json"))
    minutes = {run["run_name"]: run["minutes"] for run in durations["runs"]}
    runs = {entry["training_manifest"]["config"]["run_name"] for entry in load_json(RECORD.file("checkpoints.json")).values()}
    assert runs == set(minutes), f"training times missing or extra: {sorted(runs ^ set(minutes))}"
    return {"device": durations["device"], "minutes_min": min(minutes.values()), "minutes_max": max(minutes.values())}


def compute() -> dict:
    learned, recipe = _learned(list(load_results(RESULTS).values()), list(load_json(RECORD.file("checkpoints.json")).values()))
    baselines = _baselines(recipe)
    # Every policy views the full scene first: one first glimpse, one protocol.
    first = {curve["per_timestep"][0]["mean"] for curve in [*learned, *baselines]}
    assert len(first) == 1, f"the first glimpse differs between rows: {first}"
    seeds = Counter(len(curve["seeds"]) for curve in learned)
    return {
        "_provenance": {"ci": CONFIDENCE, "n_seeds_per_row": dict(seeds)},
        "recipe": recipe,
        "training_time": _training_time(),
        "learned": learned,
        "baselines": baselines,
    }


dataset = Dataset(name="learned_policies", compute=compute, inputs=(RESULTS, RECORD, seg.RESULTS))

PLOTTED_SELECTION = "validation"  # each run's best scorer
# Heuristic and learned policies apart, by line style, under their own legend headers.
KIND_LINESTYLE = {"heuristic": "--", "learned": "-"}
KIND_HEADER = {"heuristic": "Heuristic:", "learned": "Learned:"}


def plot(data: dict) -> MplFigure:
    """ADE20K mIoU by timestep over the learned policies' episodes, drawn as the canvas-grid figure is, with heuristic
    and learned policies apart by line style."""
    num_glimpses = data["recipe"]["num_glimpses"]
    baselines = {c["policy"]: c for c in data["baselines"]}
    learned = {c["objective"]: c for c in data["learned"] if c["selection"] == PLOTTED_SELECTION}
    groups = {  # (style, points, runs) in the table's order
        "heuristic": [
            (style, baselines[policy]["per_timestep"][:num_glimpses], baselines[policy]["n_runs"])
            for policy, style in POLICY_STYLES.items() if policy in baselines
        ],
        "learned": [
            (style, learned[objective]["per_timestep"], len(learned[objective]["seeds"]))
            for objective, style in LEARNED_POLICY_STYLES.items()
        ],
    }

    fig, axs = uplt.subplots(ncols=1, figwidth=SINGLE_PANEL_FIGWIDTH, figheight=SINGLE_PANEL_FIGHEIGHT, abc=False)
    ax = axs[0]
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.2)
    timesteps = np.arange(num_glimpses)
    for kind, curves in groups.items():
        for style, points, n_runs in curves:
            mean_and_ci(ax, timesteps, points, n_runs=n_runs, color=style["color"], linestyle=KIND_LINESTYLE[kind],
                        marker="o", markersize=3)
    ax.set_xticks(timesteps)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Timestep $t$", fontsize=LABEL_SIZE)
    ax.set_ylabel("ADE20K mIoU (%)", fontsize=LABEL_SIZE)
    bottom_legend_with_headers(fig, [
        (KIND_HEADER[kind], [(style["color"], style["label"], KIND_LINESTYLE[kind]) for style, _, _ in curves])
        for kind, curves in groups.items()
    ])
    return fig


figure = Figure(name="learned_policies_miou", plot=plot, datasets=(dataset,))
