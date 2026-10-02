# CanViT: Toward Active-Vision Foundation Models (NeurIPS 2026)

The paper sources ([arXiv:2603.22570](https://arxiv.org/abs/2603.22570)) include the LaTeX manuscript, its generated
figures and data, and the exporter that turns CanViT evaluation results into those generated inputs.

| Path | Contents |
|---|---|
| `latex/` | the manuscript, its camera-ready and preprint wrappers, the checklist, the bibliography, the NeurIPS style |
| `latex/generate_data.py` | writes `data.tex`, `data_macros.json` and the table row files from `exports/*.json` |
| `exports/` | what the LaTeX and site read: JSON data plus generated SVG, PDF and PNG figures (`latex/figures/exported` links here) |
| `exporter/` | the pipeline that writes `exports/` from evaluation results ([its README](exporter/README.md)) |

## Building the paper

From `paper/`, these recipes need [tectonic](https://tectonic-typesetting.github.io/), [uv](https://docs.astral.sh/uv/) and
[just](https://just.systems/); `pdffonts` (poppler) for the font check.

```bash
just --list                 # every recipe
just build-camera-ready     # latex/camera_ready_wrapper.pdf, the accepted-paper build
just build-preprint         # latex/preprint_wrapper.pdf, the identified-author arXiv build
just build-arxiv            # latex/arxiv_bundle.tar.gz and its checked source directory
just check-data             # fail if generated LaTeX files do not match exports/
```

Each build regenerates the data from `exports/` first.

## Regenerating the data and figures

`just exports` runs the exporter first and then regenerates the LaTeX data bindings. The exporter reads evaluation
artifacts under `exporter/data/`; those artifacts are not included in this checkout, and [the exporter README](exporter/README.md)
lists each required directory and producer. `just check-data` compares generated LaTeX files with `exports/*.json`.
The committed figures are the renders used for the paper; regeneration preserves their data while allowing small
rendering differences.
