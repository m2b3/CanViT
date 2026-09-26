"""The CanViT (Canvas Vision Transformer) project: paper, authors, code, project page, checkpoints, citation."""

TITLE = "CanViT: Toward Active-Vision Foundation Models"
DESCRIPTION = (
    "CanViT, the Canvas Vision Transformer, is an active-vision foundation model: it sees a scene through a "
    "sequence of glimpses and remembers it on a scene-wide canvas."
)
AUTHORS = ("Yohaï-Eliel Berreby", "Sabrina Du", "Audrey Durand", "B. Suresh Krishna")
VENUE = "NeurIPS 2026"
ARXIV_ID = "2603.22570"
PAPER_URL = f"https://arxiv.org/abs/{ARXIV_ID}"
CODE_URL = "https://github.com/m2b3/CanViT"
PAGE_URL = "https://m2b3.github.io/CanViT/"
HUB_ORGANIZATION = "canvit"
HUB_ORG_URL = f"https://huggingface.co/{HUB_ORGANIZATION}"
GIT_INSTALL_SPEC = f"canvit-pytorch @ git+{CODE_URL}.git#subdirectory=canvit-pytorch"

BIBTEX = r"""@article{berreby2026canvit,
  title={CanViT: Toward Active-Vision Foundation Models},
  author={Berreby, Yoha{\"i}-Eliel and Du, Sabrina and Durand, Audrey and Krishna, B. Suresh},
  year={2026},
  eprint={2603.22570},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.22570}
}"""
