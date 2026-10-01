# MAIN 2026 talk: open work

Everything still to do for the talk: Yohaï's open requests, known issues, the work behind the slides, and the
authors' decisions. It is updated in the same step a request arrives or an item lands, so nothing is lost between
sessions; a done item is deleted (git keeps it). Each slide's status is in `OUTLINE.md`; slides are named by their id
in `index.html`. Difficulty: S (hours), M (a day), L (days). The NeurIPS camera-ready is due 2026-11-06, a week after
the talk.

## Goal

[Yohaï, 2026-10-01] "Make the slides better and better and better, build tools we might need, improve the backup
slides portion, be more precise, more impactful, anticipate questions, think about the audience, keep track of
everything that might be relevant, keywords, key ideas, what is worth spending more time on, etc. Do not
over-obsess about timing for now, focus on content, logic, clarity, importance of concepts, WHY something might
matter, visual storytelling, convincing the audience, making things cool and easy and understandable, etc. You MUST
understand before you try to COMMUNICATE." Questions to anticipate and the key ideas live in `AGENTS.md`.

## Open requests

- [Yohaï, 2026-10-01] `<canvit-live>` in the deck failed: "no available backend found. ERR: [wasm] RuntimeError:
  Aborted(InternalError: out of memory)". Measured on 2026-10-01 in headless Chromium (`throwaway/live_memory.py`,
  `slide_memory.py`, `live_wasm.py`): the deck with no slide 376 MB RSS; every slide without the model 1.46 GB (static
  slides about +100 MB each alone; `#rollout` +200, `#memory` +210, `#policies` +320); the model created at page
  open (`autoload`) brought the deck to about 3.5 GB. Fixed: the deck now adds `autoload` only when `#live` is the
  current or next slide (`data-load-near`, `../deck.js`), never in the speaker view's previews; with WebGPU disabled the
  model then starts on WebAssembly in the deck (peak about 1.9 GB). Open: whether Yohaï's browser (which one, speaker
  view open?) still fails; releasing the model when the slide is left; making the rollouts load only near their slide.
- [Yohaï, 2026-10-01] `#foundation`: the DINOv3 feature map "looks [bad]. it is blurry because you must have run
  interpolation instead of nearest ... choose other principal components. it looks washed out ... rerun it at higher
  res ... we want the objects to stand out. it could also transition into becoming smaller and having arrows that go
  from the feature map to probability maps using our ade20k segmentation heads to show it is EASY to decode what's
  where from those features"; and "beware of OVERLAYS - they can confuse ... exploring both side by side / one after
  the other, and overlays". In progress: 1024 px input, PCs 2–4 clipped 2–98%, nearest neighbor
  (`throwaway/foundation`); the released DINOv3 ViT-B ADE20K probe decodes television, fireplace, armchair.
- [Yohaï, 2026-10-01] "when you show examples of canvit, contrast with simply having dinov3 128px full scene": beside
  CanViT's maps, DINOv3 ViT-B on the whole scene at 128 px (the same input budget as a glimpse; its released 128 px
  ADE20K probe, `hub.repos.released_dinov3_ade20k_probe("dv3b", input_size_px=128)`), on the rollout, extrapolation
  and detail slides; the paper's numbers for the matched comparison (passive_comparison_rows: DINOv3 ViT-B at 128 px
  28.8 mIoU against CanViT-B's 29.3 to 39.6 from one full-scene glimpse) say why it matters.
- [Yohaï, 2026-10-01] `#distillation`: replace the conference room (natively 256 px, shown upscaled; chosen for
  convenience). The sweep (`throwaway/distillation`, 2026-10-01) proposes ADE_val_00000124 (bedroom: the unseen upper
  window and the unseen side of the bed fill in), then 00001140, 00001784, 00001847; build with the sequential
  pattern of `OUTLINE.md`.
- [Yohaï, 2026-10-01] The memory slide: "why would a slide be called that" (now "A persistent, evolving understanding
  of the scene") and "the example you chose really doesn't look inspiring": a three-object example in a lively scene
  is being swept (`OUTLINE.md`, its example properties).
- [Yohaï, 2026-10-01] "Spatial coverage and perception of detail" (`#detail`): "the idea ... is good and nice but the
  way it is showed is really not good atm". Rebuilt 2026-10-01 (one example, a zoom into the model's own input);
  awaiting the authors' look.
- [Yohaï, 2026-10-01] `#uncertainty`: "metacognition" said with its scope (`sources/concepts.md`: Fleming 2024 keeps
  "sensitivity to uncertainty" apart from metacognition; Renninger, Verghese & Coughlan 2007: people fixate where
  uncertainty is highest, EG-C2F's rule); decide the wording with the authors.
- [Yohaï, 2026-10-01] Careful, step-by-step, visually supported storytelling on every slide: introduce what a visual
  needs before showing it (`../AGENTS.md`, "Writing slides"). Review every slide for it.
- [Yohaï, 2026-10-01] Reuse the well-received primitives of the September deck, ported (never imported): see
  "Components and deck primitives".
- [Yohaï, 2026-10-01] Keep every TODO, known issue and request in this file.

## Next

- [Yohaï, 2026-10-01, an idea for after the rest] Give a sense of how good DINOv3 is and how much went into it: its
  training data (LVD-1689M, 1.7 billion images), the 7B teacher it is distilled from, its compute, where it stands on
  dense benchmarks; every number read in the DINOv3 paper (Siméoni et al. 2025, arXiv:2508.10104). A build state of
  `#foundation` or a slide of its own. S–M.

- [Yohaï, 2026-10-01] `#history` and `#results` fail at their jobs: overloaded, hard to parse, not showing the right
  things; make end-to-end training against frozen features with a linear readout explicit. ImageNet-1k and ADE20K one
  after the other, never side by side from the start, after explaining classification against segmentation and why
  segmentation is harder (`AGENTS.md`, "Visual decisions"). Passive lines restricted to base-size models, like the
  active ones ("comparing to nonsensically large models is indeed stupid and counterproductive"): the base-size
  passive points are being collected and verified into `sources/sota-history.json`; then filter the chart's passive
  lines by `size_class` and fix the notes (XXX in `index.html`). Built: the named-state slide with its examples. M.
- [Yohaï, 2026-10-01] An exploration page for the history data: `_explore-history.html` (untracked) shows every point
  with series, size and numeric axes; extend it with training and inference cost once those fields exist. S.
- [Yohaï, 2026-10-01] A training-cost view ("having a 'training flops' graph could be interesting as well idk. or
  training cost"): training compute or cost of the active models and of CanViT-B, from their papers and the rebuttal
  (training glimpses, epochs, GPU-hours); what each paper reports differs, so the qualifiers stay with each number. M.
- [Yohaï, 2026-10-01] `#frontier` (accuracy against cost) must first make the case for why anyone cares about cost:
  money, cheap experiments (`AGENTS.md`, "The story"); training cost first, the main concern for scientists, then
  inference. Training: pretrained once and released; adapting it is a linear probe or a fine-tuning under 800 USD;
  its pretraining is comparable to AdaGlimpse's own, not cheaper than every prior model (`AGENTS.md`, rebuttal facts).
  Gather verified facts that make cost concrete: inference GFLOPs
  per glimpse and per rollout against AME and AdaGlimpse (`../../assets/paper/ade20k_seg.json`), wall time per glimpse
  measured on a laptop and in the browser, the paper's training costs (§6, App. H); then design how the slide shows
  the case (a build state before the chart, or the chart's cost axis translated into time or money). M.

- The cognition links (`sources/cognition.md`, `sources/concepts.md`, read 2026-10-01): trans-saccadic integration
  (Irwin 1991; Melcher 2001) on the memory slide; coarse-to-fine scene categorization in humans (Musel et al. 2012;
  Kauffmann et al. 2015) beside C2F against F2C; uncertainty-driven fixations (Renninger et al. 2007) on
  `#uncertainty`; amodal completion and boundary extension on `#table` and `#extrapolation`; as keywords and a backup
  slide with their figures. S.
- `#neuro-ai`: three lines of text on a half-empty slide; needs a visual. S–M.
- `#table`: the title wraps to two lines; a shorter written wording would keep one. S.
- Independent reviews of `OUTLINE.md` (a fresh subagent; `codex exec -m gpt-6-astra`), then the authors'. S.
- Commit the talk (staged by name; no reference to the September deck or its organization). S.

## Known issues

- The disk is nearly full (2.7 GB free at the end of 2026-10-01, shrinking from processes outside this work). Two caches
  are the user's to clear or keep: `~/.cache/uv` (18 GB; `uv cache prune` drops unreferenced entries) and
  `~/.cache/huggingface/hub` (8.1 GB, including twelve ablation checkpoints the talk does not use).
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
