"""The CanViT (Canvas Vision Transformer) project: paper, code, project page, checkpoints, citation."""

DESCRIPTION = (
    "CanViT, the Canvas Vision Transformer, is an active-vision foundation model: it sees a scene through a "
    "sequence of glimpses and remembers it on a scene-wide canvas."
)
VENUE = "NeurIPS 2026"
OPENREVIEW_ID = "cnV9GA4pKU"
PAPER_URL = f"https://openreview.net/forum?id={OPENREVIEW_ID}"
"""The paper as published at the venue."""
ARXIV_ID = "2603.22570"
ARXIV_URL = f"https://arxiv.org/abs/{ARXIV_ID}"
CODE_URL = "https://github.com/m2b3/CanViT"
PAGE_URL = "https://m2b3.github.io/CanViT/"
HUB_ORGANIZATION = "canvit"
HUB_ORG_URL = f"https://huggingface.co/{HUB_ORGANIZATION}"

LINKS = f"[Paper ({VENUE})]({PAPER_URL}) · [arXiv]({ARXIV_URL}) · [Code]({CODE_URL}) · [Project page]({PAGE_URL})"
"""The project's links, as the Markdown line its cards carry."""

DINOV3_CODE_URL = "https://github.com/facebookresearch/dinov3"
DINOV3_LICENSE_URL = "https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md"
CHECKPOINT_LICENSE = (
    f"MIT. CanViT is pretrained by distillation from [DINOv3]({DINOV3_CODE_URL}), which is released under the "
    f"[DINOv3 License]({DINOV3_LICENSE_URL})."
)
"""The license section of the card of every CanViT checkpoint and of every model built on one."""

BIBTEX = r"""@inproceedings{berreby2026canvit,
  title={{CanViT}: Toward Active-Vision Foundation Models},
  author={Berreby, Yoha{\"i}-Eliel and Du, Sabrina and Durand, Audrey and Krishna, B. Suresh},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
  url={%s},
  eprint={%s},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}""" % (PAPER_URL, ARXIV_ID)
