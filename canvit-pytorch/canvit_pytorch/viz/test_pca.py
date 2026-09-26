import numpy as np
from sklearn.decomposition import PCA

from canvit_pytorch.viz.pca import color_limits, fit_pca, layernorm, project, to_rgb


def _tokens(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(256, 64))
    return base @ rng.normal(size=(64, 64)) + rng.normal(size=64)  # correlated, off-center


def test_basis_and_projection_match_scikit_learn():
    tokens = _tokens(0)
    ours = fit_pca(tokens)
    ref = PCA(n_components=3, svd_solver="full").fit(layernorm(tokens))
    np.testing.assert_allclose(ours.components, ref.components_, atol=1e-10)
    np.testing.assert_allclose(project(ours, tokens), ref.transform(layernorm(tokens)), atol=1e-8)


def test_paper_rgb_matches_per_frame_min_max():
    tokens = _tokens(1)
    proj = project(fit_pca(tokens), tokens)
    rgb = to_rgb(proj, color_limits(proj))
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    expected = (np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1) * 255).astype(np.uint8)
    np.testing.assert_array_equal(rgb, expected)
    assert rgb.min() == 0 and rgb.max() == 254  # min-max: extremes reach both ends


def test_fixed_limits_keep_colors_comparable_across_frames():
    a, b = _tokens(2), _tokens(3)
    basis = fit_pca(b)
    pa, pb = project(basis, a), project(basis, b)
    limits = color_limits(pa, pb)
    np.testing.assert_array_equal(to_rgb(pb, limits), to_rgb(pb, color_limits(pb, pa)))
