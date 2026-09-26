# Project page

The repository's `AGENTS.md` applies here too; `README.md` documents the
components and the recording pipeline.

- The page is served under `/CanViT/`: every URL in pages and components is
  relative. `og:image` is the exception, because crawlers need it absolute.
- Read layer values with `js/canvit/png.js`, never back from a `<canvas>`:
  browsers color-manage canvas readback, and anti-fingerprinting (Brave,
  Safari private browsing, Firefox `resistFingerprinting`) perturbs it, which
  silently changes class indices.
- Maps sit beside the photograph, never over it.
- A color scale is fixed across the frames it compares (`domain` in
  `README.md`); renormalizing per frame changes what brightness means.
- Result numbers on the page come from the paper's generated macros
  (`<span data-macro>`), which `check_paper_numbers.py` verifies before every
  deploy.
