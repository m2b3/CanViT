# Paper figures

Renders of the paper's figures and tables that this talk shows, accepted with `site/slides/paper_figures.py accept`.
Each render's paper commit, source, renderer, checksum and acceptance date are its entry's `accepted` record in
`../../sources/paper-figures.json`. `paper_figures.py check`, which `site/deploy.sh` runs, fails when a file here
differs from its record or has none.
