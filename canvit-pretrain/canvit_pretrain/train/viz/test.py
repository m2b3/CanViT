"""Tests for canvit_pretrain.train.viz module."""

import numpy as np
import torch

from canvit_pretrain.train.viz.image import imagenet_denormalize_to_numpy
from canvit_pretrain.train.viz.metrics import cosine_dissimilarity
from canvit_pretrain.train.viz.pca import fit_pca, pca_rgb
from canvit_pretrain.train.viz.plot import timestep_colors


class TestFitPca:
    def test_returns_pca(self) -> None:
        features = np.random.randn(100, 64).astype(np.float32)
        pca = fit_pca(features, n_components=12)
        assert pca is not None

    def test_returns_none_for_low_variance(self) -> None:
        features = np.ones((100, 64), dtype=np.float32)
        pca = fit_pca(features, n_components=12)
        assert pca is None

    def test_clamps_components(self) -> None:
        features = np.random.randn(5, 64).astype(np.float32)
        pca = fit_pca(features, n_components=12)
        assert pca is not None
        assert pca.n_components_ == 5


class TestPcaRgb:
    def test_output_shape(self) -> None:
        features = np.random.randn(64, 128).astype(np.float32)
        pca = fit_pca(features)
        assert pca is not None
        rgb = pca_rgb(pca, features, H=8, W=8)
        assert rgb.shape == (8, 8, 3)

    def test_none_pca_returns_gray(self) -> None:
        features = np.random.randn(64, 128).astype(np.float32)
        rgb = pca_rgb(None, features, H=8, W=8)
        assert rgb.shape == (8, 8, 3)
        assert np.allclose(rgb, 0.5)


class TestImagenetDenormalizeToNumpy:
    def test_output_shape(self) -> None:
        t = torch.zeros(3, 4, 4)
        out = imagenet_denormalize_to_numpy(t)
        assert out.shape == (4, 4, 3)

    def test_output_range(self) -> None:
        t = torch.randn(3, 8, 8)
        out = imagenet_denormalize_to_numpy(t)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestCosineDissimilarity:
    def test_identical_vectors(self) -> None:
        a = np.random.randn(10, 64).astype(np.float32)
        result = cosine_dissimilarity(a, a)
        assert result.shape == (10,)
        assert np.allclose(result, 0.0, atol=1e-6)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        result = cosine_dissimilarity(a, b)
        assert np.allclose(result, 1.0, atol=1e-6)


class TestTimestepColors:
    def test_returns_correct_count(self) -> None:
        colors = timestep_colors(5)
        assert len(colors) == 5

    def test_rgba_format(self) -> None:
        colors = timestep_colors(3)
        for c in colors:
            assert len(c) == 4
