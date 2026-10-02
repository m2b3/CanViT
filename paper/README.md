# CanViT: Toward Active-Vision Foundation Models (NeurIPS 2026)

The paper's sources ([arXiv:2603.22570](https://arxiv.org/abs/2603.22570)): the LaTeX, the figures and data it
reads, and the pipeline that produces them from CanViT's evaluation results.

| Path | Contents |
|---|---|
| `latex/` | the manuscript, its camera-ready and preprint wrappers, the checklist, the bibliography, the NeurIPS style |
| `latex/generate_data.py` | writes `data.tex`, `data_macros.json` and the table row files from `exports/*.json` |
| `exports/` | what the LaTeX reads: the data as JSON and the figures as PDF (`latex/figures/exported` links here) |
| `exporter/` | the pipeline that writes `exports/` from evaluation results ([its README](exporter/README.md)) |

## Building the paper

Needs [tectonic](https://tectonic-typesetting.github.io/), [uv](https://docs.astral.sh/uv/) and
[just](https://just.systems/); `pdffonts` (poppler) for the font check.

```bash
just --list                 # every recipe
just build-camera-ready     # latex/camera_ready_wrapper.pdf
just build-preprint         # latex/preprint_wrapper.pdf, the arXiv version
just build-arxiv            # the preprint's source bundle for arXiv
just check-data             # the generated files match exports/
```

Each build regenerates the data from `exports/` first.

## Regenerating the data and figures

`just exports` runs the exporter, then regenerates the data. The exporter reads the outputs of
`python -m canvit_pytorch.evaluate` from `exporter/data/`, which this repository does not distribute; its
README lists what each dataset reads. Regenerated JSON matches `exports/` exactly. The figure PDFs in
`exports/` are the renders the paper was published with; regenerated figures carry the same data, and
their rendering can differ slightly.
