import jax
import numpy as np

from canvit_nnx.policies import MIN_SCALE, all_random, coarse_to_fine, full_then_random, quadtree_level


def test_full_then_random_starts_with_the_full_scene_and_follows_the_pretraining_law() -> None:
    viewpoints = full_then_random(jax.random.key(0), batch_size=50_000, num_glimpses=3)
    assert len(viewpoints) == 3
    np.testing.assert_array_equal(np.asarray(viewpoints[0].centers), 0.0)
    np.testing.assert_array_equal(np.asarray(viewpoints[0].scales), 1.0)
    for viewpoint in viewpoints[1:]:
        centers, scales = np.asarray(viewpoint.centers), np.asarray(viewpoint.scales)
        assert scales.min() >= MIN_SCALE and scales.max() <= 1.0
        assert np.all(np.abs(centers) + scales[:, None] <= 1.0 + 1e-6), "every crop lies inside the scene"
        # p(s) ∝ 1 - s on [MIN_SCALE, 1]: E[s] = 1 - 2/3 (1 - MIN_SCALE) and P(s <= 1/2) = 1 - (1/2 / (1 - MIN_SCALE))^2.
        assert abs(scales.mean() - (1 - 2 / 3 * (1 - MIN_SCALE))) < 5e-3
        assert abs((scales <= 0.5).mean() - (1 - (0.5 / (1 - MIN_SCALE)) ** 2)) < 1e-2
    assert not np.array_equal(np.asarray(viewpoints[1].scales), np.asarray(viewpoints[2].scales))


def test_all_random_follows_the_pretraining_law_above_its_min_scale() -> None:
    min_scale = 0.25
    viewpoints = all_random(jax.random.key(0), batch_size=50_000, num_glimpses=2, min_scale=min_scale)
    for viewpoint in viewpoints:
        centers, scales = np.asarray(viewpoint.centers), np.asarray(viewpoint.scales)
        assert scales.min() >= min_scale and np.all(np.abs(centers) + scales[:, None] <= 1.0 + 1e-6)
        assert abs(scales.mean() - (1 - 2 / 3 * (1 - min_scale))) < 5e-3
    assert not np.array_equal(np.asarray(viewpoints[0].scales), np.asarray(viewpoints[1].scales))


def test_coarse_to_fine_visits_each_quadtree_level_in_a_random_order_per_scene() -> None:
    batch_size = 64
    viewpoints = coarse_to_fine(jax.random.key(0), batch_size=batch_size, num_glimpses=21)
    np.testing.assert_array_equal(np.asarray(viewpoints[0].scales), 1.0)
    np.testing.assert_array_equal(np.asarray(viewpoints[0].centers), 0.0)
    for level, glimpses in ((1, viewpoints[1:5]), (2, viewpoints[5:21])):
        tiles = sorted(map(tuple, quadtree_level(level)))
        visits = np.stack(
            [np.concatenate([np.asarray(v.centers), np.asarray(v.scales)[:, None]], axis=1) for v in glimpses], axis=1
        )
        for scene in visits:
            assert sorted(map(tuple, scene)) == tiles, "each tile of the level once"
        assert len({tuple(scene.ravel()) for scene in visits}) > 1, "orders differ between scenes"
