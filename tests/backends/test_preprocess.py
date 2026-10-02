from typing import cast

import numpy as np
import pytest
import torch
from canvit_core.preprocess import preprocess as core_preprocess
from canvit_pytorch.preprocess import imagenet_denormalize as torch_denormalize
from canvit_pytorch.preprocess import imagenet_normalize as torch_normalize
from canvit_pytorch.preprocess import preprocess as torch_preprocess
from canvit_pytorch.preprocess import preprocess_labels as torch_preprocess_labels
from PIL import Image


def test_native_normalization_matches_pytorch_nhwc(backend):
    preprocess_module = __import__(f"canvit_{backend.name}.preprocess", fromlist=["imagenet_normalize"])
    pixels = np.random.default_rng(4040).uniform(0.0, 1.0, size=(2, 5, 7, 3)).astype(np.float32)
    torch_pixels = torch.from_numpy(pixels).permute(0, 3, 1, 2)
    expected = torch_normalize(torch_pixels).permute(0, 2, 3, 1).numpy()
    actual = backend.to_numpy(preprocess_module.imagenet_normalize(backend.array(pixels)))
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    expected_denormalized = torch_denormalize(torch_normalize(torch_pixels)).permute(0, 2, 3, 1).numpy()
    actual_denormalized = backend.to_numpy(preprocess_module.imagenet_denormalize(backend.array(expected)))
    np.testing.assert_allclose(actual_denormalized, expected_denormalized, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("height,width", [(11, 17), (10, 14), (14, 10)])
def test_native_image_and_label_preprocess_match_shared_and_torch_geometry(backend, height, width):
    preprocess_module = __import__(f"canvit_{backend.name}.preprocess", fromlist=["preprocess"])
    pixels = np.random.default_rng(4041).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    labels = np.random.default_rng(4042).integers(0, 6, size=(height, width), dtype=np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    label_image = Image.fromarray(labels, mode="L")
    expected = core_preprocess(8)(image)
    actual = preprocess_module.preprocess(8)(image)
    np.testing.assert_array_equal(actual, expected)
    torch_expected = cast(torch.Tensor, torch_preprocess(8)(image)).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(actual, torch_expected, atol=1e-6, rtol=1e-6)
    expected_labels = cast(torch.Tensor, torch_preprocess_labels(8)(label_image))[0].numpy()
    np.testing.assert_array_equal(preprocess_module.preprocess_labels(8)(label_image), expected_labels)
