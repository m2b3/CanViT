# Figure provenance

Figures from the cited works, each cited on the slide that shows it. Unless a row says otherwise, a file is a
rectangular region of the stated page rendered with poppler `pdftoppm` at the stated resolution, with no pixel edits,
then downscaled to at most 1600 px on its longer side. "p." is the printed page.

| File | Work | Figure and page | Source | Notes |
|---|---|---|---|---|
| `yamins-2014-performance-vs-it.png` | Yamins, Hong, Cadieu, Solomon, Seibert & DiCarlo, "Performance-optimized hierarchical models predict neural responses in higher visual cortex", *PNAS* 111(23):8619–8624, 2014 | Fig. 1A, p. 8620 | doi:10.1073/pnas.1403112111 | Rendered at 800 dpi. The shared x-axis label spans both panels and is omitted (the slide restates it); the panel letter is painted out in white, since the slide shows panel A alone. |
| `brain-score-2018.png` | Schrimpf et al., "Brain-Score: Which Artificial Neural Network for Object Recognition is most Brain-Like?", bioRxiv, 2018 | Fig. 1, lower half, p. 3 | doi:10.1101/407007 (version of 2018-09-05) | Brain-Score against ImageNet top-1, with the inset of models above 70% top-1. Rendered at 600 dpi. |
| `zhuang-2021-fig2b-it.png` | Zhuang, Yan, Nayebi, Schrimpf, Frank, DiCarlo & Yamins, "Unsupervised neural network models of the ventral visual stream", *PNAS* 118(3):e2014196118, 2021 (CC BY-NC-ND 4.0) | Fig. 2B, IT panel, p. 5 | doi:10.1073/pnas.2014196118 | The IT panel only; its axis title, shared with the V1 panel, is restated on the slide. Rendered at 400 dpi. |
| `raugel-2025-openreview-header.png` | Raugel et al., "Disentangling the Factors of Convergence between Brains and DINOv3", ICLR 2026 | The OpenReview page's header | The paper's OpenReview page (CC BY 4.0) | A screenshot, downscaled to 1600 px wide. |
| `raugel-2026-iclr-method.png` | as above | Method figure, p. 1 | The ICLR 2026 camera-ready on OpenReview | The PDF's embedded image with its soft mask as alpha, extracted with `pdfimages`. |
| `raugel-2026-iclr-fig4e-half-time-vs-distance.png` | as above | Fig. 4E, p. 7 | as above | The PDF's embedded image of the panel, extracted with `pdfimages`; the panel letter is not part of it. |
| `mnih-2014-fig1a-glimpse-sensor.png` | Mnih, Heess, Graves & Kavukcuoglu, "Recurrent Models of Visual Attention", NeurIPS 2014 | Fig. 1A, p. 3 | arXiv:1406.6247v1 | Panel A's box without its label. Rendered at 600 dpi. |
| `mnih-2014-fig3-glimpse-path.png` | as above | Fig. 3, first row, first column, p. 7 | as above | Rendered at 600 dpi. |
| `yarbus-1967-visitor/` | Yarbus, *Eye Movements and Vision*, Plenum Press, 1967 | Fig. 109 | https://commons.wikimedia.org/wiki/File:Yarbus_The_Visitor.jpg (public domain there; SHA-1 c9dff47eed5ccba36456dd483b2d97fc068dbf65) | Crops of the Commons file (left, top, right, bottom, in px): `painting` (6, 3, 250, 230); `record-1` (258, 3, 524, 170); `record-2` (550, 3, 766, 167); `record-3` (20, 248, 250, 400); `record-4` (272, 252, 500, 400); `record-5` (552, 258, 760, 392); `record-6` (6, 452, 252, 618); `record-7` (298, 476, 528, 616). The printed record numbers 1, 2, 6 and 7 are painted over with the page's gray (236, 236, 236) at (506, 148, 522, 168), (757, 143, 773, 164), (244, 600, 258, 620) and (518, 598, 534, 618); no traces cross those boxes. |

Generated data (`../../../../data/talk/`, deployed, not committed) also draws on third-party media:

| Data | Work | Source | Notes |
|---|---|---|---|
| `canvit2-glimpses/` | "Hippy Flip on arms", BoarderleinTV (CC BY 3.0) | https://commons.wikimedia.org/wiki/File:Hippy_Flip_on_arms.webm | `#next`'s video, 0.33 s to 3.2 s, cropped to remove the watermark and border; its manifest records the license. |
