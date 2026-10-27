"""Helpers for canvit-nnx's own tests; pytest puts this directory on sys.path through conftest.py."""

import numpy as np


def relative_l2(reference: object, actual: object) -> float:
    """||actual - reference|| / ||reference||, in float64."""
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    return float(np.linalg.norm((actual - reference).ravel()) / max(np.linalg.norm(reference.ravel()), 1e-30))
