# MAIN 2026 talk: open work

Everything still to do for the talk: Yohaï's open requests, known issues, the work behind the slides, and the
authors' decisions. It is updated in the same step a request arrives or an item lands, so nothing is lost between
sessions; a done item is deleted (git keeps it). Each slide's status is in `OUTLINE.md`; slides are named by their id
in `index.html`. Difficulty: S (hours), M (a day), L (days). The NeurIPS camera-ready is due 2026-11-06, a week after
the talk.

## Open requests

- [Yohaï, 2026-10-01] `#policies`: a race between the policies: one line per policy, in the paper's colors, growing
  glimpse by glimpse as the six animations play, against the ground truth. Plan: the paper's ADE20K mIoU per policy
  and glimpse (`../../assets/paper/ade20k_seg.json`, `policy_curves`; all of ADE20K validation), labeled as such; the
  bundles' own per-glimpse pixel accuracy on the street scene separates the policies too little (87–89% for four of
  six). S–M.
- [Yohaï, 2026-10-01] `<canvit-live>` in the deck failed: "no available backend found. ERR: [wasm] RuntimeError:
  Aborted(InternalError: out of memory)". Measured on 2026-10-01 in headless Chromium (`throwaway/live_memory.py`,
  `slide_memory.py`, `live_wasm.py`): the deck with no slide 376 MB RSS; every slide without the model 1.46 GB (static
  slides about +100 MB each alone; `#rollout` +200, `#memory` +210, `#policies` +320); the model created at page
  open (`autoload`) brought the deck to about 3.5 GB. Fixed: the deck now adds `autoload` only when `#live` is the
  current or next slide (`data-load-near`, `../deck.js`), never in the speaker view's previews; with WebGPU disabled the
  model then starts on WebAssembly in the deck (peak about 1.9 GB). Open: whether Yohaï's browser (which one, speaker
  view open?) still fails; releasing the model when the slide is left; making the rollouts load only near their slide.
- [Yohaï, 2026-10-01] Reuse the well-received primitives of the September deck, ported (never imported): see
  "Components and deck primitives".
- [Yohaï, 2026-10-01] Keep every TODO, known issue and request in this file.

## Next

- [Yohaï, 2026-10-01] `#history` and `#results` fail at their jobs: overloaded, hard to parse, not showing the right
  things; make end-to-end training against frozen features with a linear readout explicit. A timeline slide of the
  key works of deep active vision before them, from RAM, each with what it did and why it mattered (facts being
  verified: `sources/active-vision-timeline.md`), in the spirit of Yohaï's typst timeline
  (github.com/yberreby/typst-snippets, `timeline.typ`: a year axis, dots at true dates, labels above and below on
  dashed leaders). M.
- [Yohaï, 2026-10-01] Zoom and detail (`#detail`, now in Backup): try several presentations (segmentation instead of a
  probability blob, the full-scene glimpse with a loupe on the object, recall against object size with the
  full-scene-twice control, an animated zoom), render them, integrate. M.

- The cognition citation on `#table` (`sources/cognition.md`: Intraub & Richardson 1989; Biederman, Mezzanotte &
  Rabinowitz 1982; Torralba et al. 2006) and a backup slide with their figures (`assets/figures/`, extracted). Prepare
  the objection that edge continuation fills the gap without knowledge of tables (amodal completion). S.
- `#neuro-ai`: three lines of text on a half-empty slide; needs a visual. S–M.
- `#table`: the title wraps to two lines; a shorter written wording would keep one. S.
- Independent reviews of `OUTLINE.md` (a fresh subagent; `codex exec -m gpt-6-astra`), then the authors'. S.
- Commit the talk (staged by name; no reference to the September deck or its organization). S.

## Known issues

- The disk is nearly full (about 250 MB free on 2026-10-01, shrinking from processes outside this work). Two caches
  are the user's to clear or keep: `~/.cache/uv` (18 GB; `uv cache prune` drops unreferenced entries) and
  `~/.cache/huggingface/hub` (8.1 GB, including twelve ablation checkpoints the talk does not use).
- `#memory`: `<canvit-path>` is shrunk with `zoom: .72` to fit; a slide-sized layout of the component would be cleaner.
- `#live` (Backup): shrunk with `zoom: .8`.
- `assets/figures/`: 21 MB, many figures unused by the current slides; downscale the used ones to slide size and leave
  the rest uncommitted.

## Experiments and exports

| Item | Slides | Diff. | Also | Status |
|---|---|---|---|---|
| Table corners: two glimpses at an object's ends, CanViT against DINOv3 per glimpse, over ADE20K validation (`throwaway/table_corners`); to graduate into `canvit_pytorch.viz` | `#table`, `#extrapolation` | S | page | done |
| Distillation: DINOv3's features of a scene and CanViT's prediction after each glimpse (`throwaway/distillation`) | `#distillation` | S | page, paper | done |
| Looking closer: small objects missed by a full-scene glimpse and found by a zoomed one (`throwaway/looking_closer`); to graduate into `canvit_pytorch.viz` | `#detail` | S | page | done |
| Teacher features of the conference room for the browser (similarity to a hovered patch, PCA, probe segmentation) | `#foundation` | M | page | todo |

## Components and deck primitives

| Item | Slides | Diff. | Status |
|---|---|---|---|
| `shoot.py` checks: titles that wrap, content in the footer band (below 676 px), single words alone on a last line | all | S | todo |
| The table demonstration animated: each glimpse's passive answer flying to its place in the scene map (stepped CSS, `--x0/--y0/--s0` to `--x1/--y1/--s1`) | `#table` | M | todo |
| `<canvit-foveate>`: glimpses jumping over a periphery blurred by eccentricity, which fades to nothing on a click | `#human-vision`, `#glimpses` | M | todo |
| `<canvit-features>`: similarity to the hovered patch, PCA colors, the probe's segmentation | `#foundation` | M | todo |
| The architecture built click by click, pulses along read and write arrows (SVG `animateMotion`), real tokens from a bundle | `#architecture` | M | todo |
| `<canvit-live>` laid out for a slide; one model session for every live slide | `#live` | M | todo |
| Canvases sized from their displayed size, sharp when the deck scales (episode, mosaic) | all | S | todo |

## Logistics

| Item | Diff. | Status |
|---|---|---|
| Offline on the day: ONNX Runtime Web loads from jsDelivr (`js/canvit/live-model.js`); install it locally as reveal.js is | S–M | todo |

## For the camera-ready (not the talk)

- Remove "an order of magnitude more than previous active models" from the abstract [Yohaï, 2026-10-01]; the project
  page's abstract and the README copy it and follow.

## Decisions for the authors

- LookWhere (NeurIPS 2025; 83.0 ImageNet-1k, 44.6 ADE20K, patches selected once from a low-resolution view): address
  it in the talk, or only if asked.
- Learned viewing policies (CanViT-PyTorch-RL, unpublished): show them or not.
- The funders on the last slide: those of the paper's acknowledgments, confirmed.
- Sabrina Du's portrait (`../../assets/authors/du.jpg`) was provided privately: confirm she agrees to its
  publication before the branch is pushed.
- Who presents what, if Sabrina co-presents.
- JAX and MLX ports: clean them up for "runs everywhere", or say PyTorch and the browser.
