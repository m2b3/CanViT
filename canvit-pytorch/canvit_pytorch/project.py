"""The CanViT (Canvas Vision Transformer) project: paper, code, project page, checkpoints, citation."""

DESCRIPTION = (
    "CanViT, the Canvas Vision Transformer, is an active-vision foundation model: it sees a scene through a "
    "sequence of glimpses and remembers it on a scene-wide canvas."
)
VENUE = "NeurIPS 2026"
ARXIV_ID = "2603.22570"
PAPER_URL = f"https://arxiv.org/abs/{ARXIV_ID}"
CODE_URL = "https://github.com/m2b3/CanViT"
PAGE_URL = "https://m2b3.github.io/CanViT/"
HUB_ORGANIZATION = "canvit"
HUB_ORG_URL = f"https://huggingface.co/{HUB_ORGANIZATION}"

DINOV3_CODE_URL = "https://github.com/facebookresearch/dinov3"
DINOV3_LICENSE_URL = "https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md"
CHECKPOINT_LICENSE = (
    f"MIT. CanViT is pretrained by distillation from [DINOv3]({DINOV3_CODE_URL}), which is released under the "
    f"[DINOv3 License]({DINOV3_LICENSE_URL})."
)
"""The license section of the card of every CanViT checkpoint and of every model built on one."""

BIBTEX = r"""@inproceedings{berreby2026canvit,
  title={CanViT: Toward Active-Vision Foundation Models},
  author={Berreby, Yoha{\"i}-Eliel and Du, Sabrina and Durand, Audrey and Krishna, B. Suresh},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  eprint={2603.22570},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.22570}
}"""
