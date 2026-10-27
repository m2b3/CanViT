"""Each viewing policy's label and line color, so that the tables mark policies as the figures draw them."""

from canvit_paper_exporter.core import Dataset
from canvit_paper_exporter.style import LEARNED_POLICY_STYLES, POLICY_STYLES


def compute() -> dict:
    return {"policies": POLICY_STYLES, "learned": LEARNED_POLICY_STYLES}


dataset = Dataset(name="policy_styles", compute=compute)
