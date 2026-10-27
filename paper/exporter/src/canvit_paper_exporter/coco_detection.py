"""COCO val2017 AP under C2F of CanViT fine-tuned with a CenterNet detection head, from the JAX/Flax NNX implementation's
evaluation records: the two models of the architecture control read at the same training step, and the paper's
ImageNet-21k CanViT-B fine-tuned with a longer horizon and a larger batch."""

from canvit_pytorch.hub.repos import FLAGSHIP

from canvit_paper_exporter.core import Dataset, load_json
from canvit_paper_exporter.paths import Inputs

MATCHED_STEP = 16_000
MATCHED = {
    "canvit": "matched/cv-dv3_MATCHED_s16000_n5000_r5_SMOKE.json",
    "adaglimpse": "matched/ag-dv3_MATCHED_s16000_n5000_r5_SMOKE.json",
}
FLAGSHIP_EVALUATION = "t10_b1024_2026-08-01/cv-f21k_BEST_s16172_n5000_r5_t21_20260801T231033Z.json"
FLAGSHIP_TRAINING = "t10_b1024_2026-08-01/coco-cv-f21k-e140-b1024-t10-v3.json"
REPEATS = 5
TRAIN_IMAGES = 118_287  # COCO train2017, the fine-tuning set
TRAINING_KEYS = ("n_timesteps", "batch_size", "peak_lr", "backbone_lr", "weight_decay")
RECORDS = Inputs("coco_detection", (*MATCHED.values(), FLAGSHIP_EVALUATION, FLAGSHIP_TRAINING))


def _evaluation(name: str) -> dict:
    """An evaluation of every val2017 image (eval_batches 0), REPEATS times under C2F."""
    record = load_json(RECORDS.file(name))
    conditions = (record["eval_only"], record["eval_policy"], record["eval_batches"], record["eval_repeats"])
    assert conditions == (True, "c2f", 0, REPEATS), (name, conditions)
    return record


def _ap(record: dict) -> list[dict]:
    final = record["final"]
    return [
        {"t": t, "ap": final[f"AP_c2f_t{t}"], "ci_lo": final[f"AP_c2f_t{t}_ci_lo"], "ci_hi": final[f"AP_c2f_t{t}_ci_hi"]}
        for t in range(record["n_timesteps"])
    ]


def _geometry(record: dict) -> tuple[int, int]:
    """The evaluation's scene size (px) and canvas grid."""
    return record["scene_px"], record["canvas_grid"]


def _matched(arm: str) -> dict:
    record = _evaluation(MATCHED[arm])
    assert record["arm"] == arm and record["final"]["step"] == MATCHED_STEP, (arm, record["final"]["step"])
    training = {k: record[k] for k in TRAINING_KEYS} | {"steps": record["max_steps"]}
    return {"head_width": record["head_width"], "training": training, "geometry": _geometry(record), "per_timestep": _ap(record)}


def _flagship() -> dict:
    evaluation, training = _evaluation(FLAGSHIP_EVALUATION), load_json(RECORDS.file(FLAGSHIP_TRAINING))
    assert evaluation["hf_repo"] == training["hf_repo"] == FLAGSHIP, (evaluation["hf_repo"], training["hf_repo"])
    assert evaluation["restore_step"] == training["max_steps"], "the evaluation must score the last training step"
    keys = ("n_timesteps", "batch_size", "max_steps", "warmup_steps", "peak_lr", "backbone_lr", "weight_decay",
            "backbone_weight_decay", "head_width", "min_view_scale", "train_policy", "scene_px", "canvas_grid", "glimpse_px")
    return {
        "model_repo": FLAGSHIP, "training": {k: training[k] for k in keys}, "geometry": _geometry(evaluation),
        "per_timestep": _ap(evaluation),
    }


def compute() -> dict:
    arms, flagship = {arm: _matched(arm) for arm in MATCHED}, _flagship()
    geometries = {flagship.pop("geometry"), *(arm.pop("geometry") for arm in arms.values())}
    assert len(geometries) == 1, f"the models were evaluated on different scenes or canvases: {geometries}"
    [(scene_px, canvas_grid)] = geometries
    return {
        "_provenance": {"policy": "c2f", "repeats": REPEATS, "ci": 0.95},
        "train_images": TRAIN_IMAGES,
        "scene_px": scene_px,
        "canvas_grid": canvas_grid,
        "matched": {"step": MATCHED_STEP, "arms": arms},
        "flagship": flagship,
    }


dataset = Dataset(name="coco_detection", compute=compute, inputs=(RECORDS,))
