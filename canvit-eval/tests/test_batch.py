"""Pure tests for the batch eval matrix — no GPU/data needed."""
import pytest
from pathlib import Path

from canvit_pytorch.checkpoints import (
    ABLATION_CHECKPOINTS,
    ABLATION_MODEL_SHORTS,
    PRETRAIN_CHECKPOINTS,
    ade20k_probe_repo,
)

from canvit_eval.batch import (
    DETERMINISTIC,
    DEFAULT_TASKS,
    CANVAS_GRIDS,
    EXTRA_CANVAS_GRIDS,
    EXTRA_IN1K_RESOLUTIONS,
    IN1K_RESOLUTIONS,
    _apply_shard,
    build_eval_matrix,
    filter_jobs,
)


def test_breadth_first_scheduling():
    """Every r=0 runs before any r=1: an interrupted batch yields n=1 per cell."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=5, n_timesteps=21, tasks=DEFAULT_TASKS)
    positions_by_run_idx: dict[int, list[int]] = {}
    for i, j in enumerate(jobs):
        positions_by_run_idx.setdefault(j.run_idx, []).append(i)
    for r in sorted(positions_by_run_idx)[:-1]:
        assert max(positions_by_run_idx[r]) < min(positions_by_run_idx[r + 1]), (
            f"run_idx {r + 1} starts before run_idx {r} finishes — not breadth-first"
        )


def test_eval_job_structural_tuple_is_unique():
    """(task, model, policy, scene_size, canvas_grid, input_px, run_idx) is the
    skip-existing matching key — must be unique per job."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=10, n_timesteps=21, tasks=DEFAULT_TASKS,
                             include_extra_grids=True)
    keys = [(j.task, j.model, j.policy, j.scene_size, j.canvas_grid, j.input_px, j.run_idx)
            for j in jobs]
    assert len(keys) == len(set(keys))


def test_eval_job_output_path_uniqueness():
    """Within one matrix build (single timestamp), no two jobs share an output path."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=10, n_timesteps=21, tasks=DEFAULT_TASKS,
                             include_extra_grids=True)
    paths = [j.output for j in jobs]
    assert len(paths) == len(set(paths))


def test_extra_canvas_grids_disjoint_from_canvas_grids():
    """(512, 8) and (512, 16) are in CANVAS_GRIDS — duplicating into EXTRA_*
    would produce identical t=0 output paths."""
    assert set(EXTRA_CANVAS_GRIDS).isdisjoint(set(CANVAS_GRIDS))


def test_in1k_resolutions_disjoint():
    """Baseline vs extras must not overlap on (scene, grid)."""
    baseline = {(s, g) for s, g, _ in IN1K_RESOLUTIONS}
    extras = {(s, g) for s, g, _ in EXTRA_IN1K_RESOLUTIONS}
    assert baseline.isdisjoint(extras)


def test_entropy_c2f_skipped_on_non_power_of_two():
    """entropy_coarse_to_fine partitions the canvas into 2x2 / 4x4 tiles; only
    power-of-two grids align. Non-pow2 grids must be excluded; pow2 extras must
    still get the policy."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21,
                             tasks=["ade20k-seg"], include_extra_grids=True)
    non_pow2 = {g for _, g in EXTRA_CANVAS_GRIDS if (g & (g - 1)) != 0}
    leaked = [j for j in jobs if j.policy == "entropy_coarse_to_fine" and j.canvas_grid in non_pow2]
    assert leaked == [], f"entropy_c2f leaked into non-pow2 grids: {leaked}"
    pow2_extra = {g for _, g in EXTRA_CANVAS_GRIDS if (g & (g - 1)) == 0}
    if pow2_extra:
        on_pow2 = [j for j in jobs if j.policy == "entropy_coarse_to_fine" and j.canvas_grid in pow2_extra]
        assert on_pow2, "entropy_c2f missing on power-of-2 extras"


def test_dinov3_canvas_grid_derivation():
    """DINOv3 jobs carry canvas_grid = input_px // 16 (patch size)."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    dv3 = [j for j in jobs if j.model.startswith("dinov3-")]
    assert dv3
    for j in dv3:
        assert j.input_px is not None and j.canvas_grid is not None
        assert j.canvas_grid == j.input_px // 16


def test_in1k_filename_encodes_scene_and_grid():
    """IN1k outputs carry s{scene}_c{grid} so the exporter can group runs."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["in1k-clf"])
    in1k = [j for j in jobs if j.task == "in1k-clf"]
    assert in1k
    for j in in1k:
        assert j.scene_size is not None and j.canvas_grid is not None
        assert f"s{j.scene_size}_c{j.canvas_grid}" in j.output.name


def test_in1k_scene_and_grid_appear_in_cli_args():
    """--scene-size, --batch-size, --episode.canvas-grid must reach the task; without
    them, defaults silently override (risk: OOM at large grids)."""
    bs_lookup = {(s, g): bs for s, g, bs in IN1K_RESOLUTIONS + EXTRA_IN1K_RESOLUTIONS}
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21,
                             tasks=["in1k-clf"], include_extra_grids=True)
    for j in jobs:
        if j.task != "in1k-clf":
            continue
        for flag, expected in [
            ("--scene-size", str(j.scene_size)),
            ("--episode.canvas-grid", str(j.canvas_grid)),
            ("--batch-size", str(bs_lookup[(j.scene_size, j.canvas_grid)])),
        ]:
            assert flag in j.args, f"{flag} missing from {j.args}"
            assert j.args[j.args.index(flag) + 1] == expected


def test_in1k_extras_only_expand_frozen():
    """Finetuned weights were specialised at one (scene, grid); extras shouldn't multiply
    finetuned jobs."""
    base = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["in1k-clf"])
    ext = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21,
                            tasks=["in1k-clf"], include_extra_grids=True)
    base_frozen = [j for j in base if j.model == "canvit-frozen"]
    ext_frozen = [j for j in ext if j.model == "canvit-frozen"]
    base_ft = [j for j in base if j.model == "canvit-finetuned"]
    ext_ft = [j for j in ext if j.model == "canvit-finetuned"]
    assert len(ext_frozen) > len(base_frozen)
    assert len(ext_ft) == len(base_ft)


def test_filter_by_policy_drops_dinov3():
    """DINOv3 jobs have policy=None; a policy filter must drop them."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    kept = filter_jobs(jobs, policies=["coarse_to_fine"])
    assert all(j.policy == "coarse_to_fine" for j in kept)
    assert not any(j.model.startswith("dinov3-") for j in kept)


def test_filter_by_grid():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    kept = filter_jobs(jobs, grids=[32])
    assert all(j.canvas_grid == 32 for j in kept)


def test_skip_existing_is_timestamp_agnostic(tmp_path):
    """already_done() globs on the structural stem; matches any prior-run timestamp."""
    ade_dir = tmp_path / "results" / "ade20k_seg"
    ade_dir.mkdir(parents=True)
    (ade_dir / "coarse_to_fine_s512_c32_20260101T000000Z_r0.pt").touch()

    jobs = build_eval_matrix(tmp_path / "results", n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    target = next(j for j in jobs if j.policy == "coarse_to_fine"
                  and j.scene_size == 512 and j.canvas_grid == 32 and j.run_idx == 0
                  and j.input_px is None)
    assert target.already_done()


def test_skip_existing_is_per_run_idx(tmp_path):
    """Existing r=0 must NOT satisfy r=1's check; different run_idx are independent samples."""
    ade_dir = tmp_path / "results" / "ade20k_seg"
    ade_dir.mkdir(parents=True)
    (ade_dir / "coarse_to_fine_s512_c32_20260101T000000Z_r0.pt").touch()

    jobs = build_eval_matrix(tmp_path / "results", n_runs=2, n_timesteps=21, tasks=["ade20k-seg"])
    r0 = next(j for j in jobs if j.policy == "coarse_to_fine"
              and j.canvas_grid == 32 and j.scene_size == 512 and j.run_idx == 0
              and j.input_px is None)
    r1 = next(j for j in jobs if j.policy == "coarse_to_fine"
              and j.canvas_grid == 32 and j.scene_size == 512 and j.run_idx == 1
              and j.input_px is None)
    assert r0.already_done()
    assert not r1.already_done()


def test_skip_existing_respects_policy_scene_grid(tmp_path):
    """Existing data at (c2f, s512, c32) must NOT satisfy other policies / scenes / grids."""
    ade_dir = tmp_path / "results" / "ade20k_seg"
    ade_dir.mkdir(parents=True)
    (ade_dir / "coarse_to_fine_s512_c32_20260101T000000Z_r0.pt").touch()

    jobs = build_eval_matrix(tmp_path / "results", n_runs=1, n_timesteps=21,
                             tasks=["ade20k-seg"], include_extra_grids=True)
    other_policy = next(j for j in jobs if j.policy == "fine_to_coarse"
                        and j.canvas_grid == 32 and j.scene_size == 512)
    other_grid = next(j for j in jobs if j.policy == "coarse_to_fine"
                      and j.canvas_grid == 9 and j.scene_size == 512)
    other_scene = next(j for j in jobs if j.policy == "coarse_to_fine"
                       and j.canvas_grid == 64 and j.scene_size == 1024)
    assert not other_policy.already_done()
    assert not other_grid.already_done()
    assert not other_scene.already_done()


def test_skip_existing_fills_partial_n_runs(tmp_path):
    """Existing r=0..2 + --n-runs 5 leaves r=0..2 in place and queues r=3, r=4."""
    ade_dir = tmp_path / "results" / "ade20k_seg"
    ade_dir.mkdir(parents=True)
    for run in range(3):
        (ade_dir / f"coarse_to_fine_s512_c32_20260101T000000Z_r{run}.pt").touch()

    jobs = build_eval_matrix(tmp_path / "results", n_runs=5, n_timesteps=21, tasks=["ade20k-seg"])
    cells = [j for j in jobs if j.policy == "coarse_to_fine"
             and j.scene_size == 512 and j.canvas_grid == 32 and j.input_px is None]
    assert len(cells) == 5
    done = sorted(j.run_idx for j in cells if j.already_done())
    pending = sorted(j.run_idx for j in cells if not j.already_done())
    assert done == [0, 1, 2]
    assert pending == [3, 4]


def test_ablation_seg_pairs_model_and_probe_by_slug():
    """The seg CLI defaults --episode.model-repo to the flagship, and most
    ablation variants share its canvas_dim — an ablation probe against the
    wrong model would pass the embed-dim assert and produce plausible wrong
    numbers. Every job must pin the model explicitly and pair it with the
    probe published for that same checkpoint."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=10, n_timesteps=21,
                             tasks=["ade20k-seg-ablations"])
    # 12 variants x (4 stochastic x 10 runs + 2 deterministic x 1) at (512, 32).
    assert len(jobs) == 12 * (4 * 10 + 2)
    for j in jobs:
        model_repo = j.args[j.args.index("--episode.model-repo") + 1]
        probe_repo = j.args[j.args.index("--probe-repo") + 1]
        slug = next(s for s, r in ABLATION_CHECKPOINTS.items() if r == model_repo)
        assert j.model == f"abl-{slug}"
        # Published-name contract: probes live under these exact repo ids.
        assert probe_repo.endswith(f"-abl-{slug}")
        assert probe_repo == ade20k_probe_repo(ABLATION_MODEL_SHORTS[model_repo], scene=512, grid=32)
        assert j.output.parent.name == "ade20k_seg_ablations"


def test_in1k_clf_ablation_jobs_pin_model_and_use_frozen_fused_head():
    """C2F-only, frozen mode, model pinned per variant; the IN1k head is the
    shared fused DINOv3 probe (default --probe-repo), so model is the only thing
    that varies. The CLI defaults the model to the flagship, so the pin matters."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=10, n_timesteps=21,
                             tasks=["in1k-clf-ablations"])
    assert len(jobs) == 12 * 10  # 12 variants x n=10 (C2F is stochastic)
    for j in jobs:
        assert j.policy == "coarse_to_fine"
        assert "--mode" in j.args and j.args[j.args.index("--mode") + 1] == "frozen"
        model_repo = j.args[j.args.index("--episode.model-repo") + 1]
        slug = next(s for s, r in ABLATION_CHECKPOINTS.items() if r == model_repo)
        assert j.model == f"abl-{slug}"
        # No per-model probe: the shared DINOv3 probe is the task default.
        assert "--probe-repo" not in j.args
        assert j.output.parent.name == "in1k_clf_ablations"
        assert j.scene_size == 512 and j.canvas_grid == 32


def test_in1k_clf_pretrain_covers_only_in21k_and_in1k():
    """The dataset-scale comparison (IN21k flagship vs IN1k-only) must pin exactly
    those two pretraining checkpoints and EXCLUDE sa1b (also in PRETRAIN_CHECKPOINTS
    but abandoned / different scene resolution). Frozen, fused shared DINOv3 probe."""
    n = 4
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=n, n_timesteps=21,
                             tasks=["in1k-clf-pretrain"])
    # 2 datasets x (4 stochastic x n + 1 deterministic RFS x 1).
    assert len(jobs) == 2 * (4 * n + 1)
    pinned = {j.args[j.args.index("--episode.model-repo") + 1] for j in jobs}
    assert pinned == {PRETRAIN_CHECKPOINTS["in21k"], PRETRAIN_CHECKPOINTS["in1k"]}
    assert PRETRAIN_CHECKPOINTS["sa1b"] not in pinned  # explicit: sa1b is out of scope
    for j in jobs:
        assert j.args[j.args.index("--mode") + 1] == "frozen"
        assert "--probe-repo" not in j.args  # shared DINOv3 probe is the default
        assert j.output.parent.name == "in1k_clf_pretrain"
        assert j.scene_size == 512 and j.canvas_grid == 32


def test_ade20k_seg_pretrain_pairs_model_and_own_probe():
    """Seg (unlike clf) has no shared fused head — each model needs its OWN ADE20K probe.
    The IN21k-vs-IN1k comparison must pin each model AND probe-ade20k-40k-s512-c{grid}-{short}
    at grids 32 and 64, and exclude sa1b."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=5, n_timesteps=21,
                             tasks=["ade20k-seg-pretrain"])
    assert {j.model for j in jobs} == {"pretrain-in21k", "pretrain-in1k"}
    assert {j.canvas_grid for j in jobs} == {32, 64}
    for j in jobs:
        short = "in1k" if j.model == "pretrain-in1k" else "in21k"
        model_repo = j.args[j.args.index("--episode.model-repo") + 1]
        probe_repo = j.args[j.args.index("--probe-repo") + 1]
        assert model_repo == PRETRAIN_CHECKPOINTS[short]
        assert probe_repo == ade20k_probe_repo(short, scene=512, grid=j.canvas_grid)
        assert j.output.parent.name == "ade20k_seg_pretrain"
    assert all(PRETRAIN_CHECKPOINTS["sa1b"] not in j.args for j in jobs)


def test_ablation_seg_deterministic_policies_run_once():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=10, n_timesteps=21,
                             tasks=["ade20k-seg-ablations"])
    runs_per_cell: dict[tuple[str, str | None], int] = {}
    for j in jobs:
        runs_per_cell[(j.model, j.policy)] = runs_per_cell.get((j.model, j.policy), 0) + 1
    for (_, policy), n in runs_per_cell.items():
        assert n == (1 if policy in DETERMINISTIC else 10)


def test_shards_partition_the_job_list():
    """Every job lands in exactly one shard; the shards' union is the whole list."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=10, n_timesteps=21,
                             tasks=["ade20k-seg-ablations"])
    for n in (1, 3, 12, 504, 1000):
        shards = [_apply_shard(jobs, f"{k}/{n}") for k in range(n)]
        recombined = [j for shard in shards for j in shard]
        assert sorted(id(j) for j in recombined) == sorted(id(j) for j in jobs)
        # Strided partition: shard sizes differ by at most 1.
        sizes = [len(s) for s in shards]
        assert max(sizes) - min(sizes) <= 1


def test_shard_rejects_bad_spec():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    for bad in ("3/3", "5/3", "-1/4", "0/0"):
        with pytest.raises((AssertionError, ValueError)):
            _apply_shard(jobs, bad)
