import pytest
import torch

from canvit_pytorch.evaluate.tasks.reconstruction import cosine_similarities
from canvit_pytorch.model.standardizer import PositionAwareStandardizer


@pytest.mark.parametrize(("positions", "target_shape"), [(1, (4, 8)), (16, (4, 16, 8))])  # CLS token, patches
def test_a_perfect_prediction_scores_one_in_both_spaces(positions: int, target_shape: tuple[int, ...]) -> None:
    """The raw score destandardizes the prediction before comparing it with the raw teacher features."""
    generator = torch.Generator().manual_seed(0)
    standardizer = PositionAwareStandardizer(positions, 8)
    standardizer.fit(torch.randn(64, positions, 8, generator=generator) * 3 + 5)
    raw_target = torch.randn(target_shape, generator=generator) * 3 + 5
    perfect = standardizer(raw_target)
    norm, raw = cosine_similarities(prediction=perfect, raw_target=raw_target, standardizer=standardizer)
    assert norm == pytest.approx(1.0) and raw == pytest.approx(1.0)
