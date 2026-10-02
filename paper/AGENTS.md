# The paper: conventions

`CLAUDE.md` is a symlink to this file. The repository's guide (`../AGENTS.md`) applies here too.

- Read `latex/CanViT_Toward_AVFMs.tex` in full before editing it or stating what the paper says.
- Build through the justfile; a bare `tectonic` skips the data step and renders stale rows.
- Never modify `latex/neurips_2026.sty`; it is the official template, byte for byte.
- Tag every manual spacing or layout tweak with `% LAYOUT_OVERRIDE:` and its reason.
- Numbers about CanViT come from `latex/data.tex`, generated from `exports/`; numbers cited from other
  papers may be written in the source.
- Generated files are never edited by hand: `exports/`, `latex/data.tex`, `latex/data_macros.json` and the
  `*_rows.tex` files. Change the exporter or `latex/generate_data.py` and regenerate.
- The project page and the talks read `latex/data_macros.json` and `exports/*.json` through links in
  `site/assets/paper/`. After changing what `generate_data.py` writes, run `uv run --no-sync --project canvit-pytorch python site/check_paper_numbers.py`
  from the repository root; before dropping a macro, also search the branches that edit `site/`
  (`git grep -w <macro> <branch>`).
- Read generated row files with `\inputrows`, which avoids the trailing `\par` of `\input` that breaks
  `\bottomrule` in tabulars.
- Acronyms go through `acro`: `\ac{AVFM}`.
- Figures are the exporter's native PDFs, read through `latex/figures/exported`.
- "Backbone" is the internal Vision Transformer, "CanViT" the full architecture, "canvas" its scene-wide
  workspace, "Viewpoint Encoding" the VPE token. Top-down feedback needs an earlier, higher-level state;
  interaction within one step is lateral (cross-stream).
- Refer to sections, figures and tables by label, never by number.
