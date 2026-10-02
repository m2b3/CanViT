# MAIN 2026 talk: open work

Everything still to do for the talk: Yohaï's open requests, known issues, the work behind the slides, and the
authors' decisions. It is updated in the same step a request arrives or an item lands, so nothing is lost between
sessions; a done item is deleted (git keeps it). Each slide's status is in `OUTLINE.md`; slides are named by their id
in `index.html`. Difficulty: S (hours), M (a day), L (days). The NeurIPS camera-ready is due 2026-11-06, a week after
the talk.

## Goal

[Yohaï, 2026-10-01] Keep improving the slides and the backup section; build the tools they need; be precise and
impactful; anticipate questions; think about the audience; track keywords, key ideas and what deserves more time.
Content, logic, clarity, the importance of each concept and why it matters, visual storytelling and convincing the
audience come before timing. Understand before communicating. Facts and scoped claims live in `sources/checked-facts.md`.

## Open requests

- [Yohaï, 2026-10-23] The talk is 25 minutes plus 5 minutes of questions. Redesign the outline around what is
  important and why, treating the existing AI-drafted slides as possibly unusable.
- [Yohaï, 2026-10-01] `#architecture`: show the intermediate glimpse stream on the architecture drawing, aligned with
  the Vision Transformer blocks (the end-of-glimpse features are enough); reads and writes in their real interleaved
  order. Rebuilt 2026-10-01 (`OUTLINE.md`); awaiting the authors' look.
- [Yohaï, 2026-10-01] `#policy-accuracy`: on ImageNet-1k, C2F's label should sit above F-IID's. The two end level
  (81.13 against 81.14% top-1 after 21 glimpses); labels now stack by end value to 0.1 point, ties going to the higher
  mean over the glimpses, which puts C2F above.
- [Yohaï, 2026-10-01] `#policies`: introduce the viewing policies one after the other rather than all moving at once,
  without their ADE20K results, which other slides cover. Rebuilt 2026-10-01: one policy per click on one rollout of
  the street, the policies so far listed beside it; no results. Awaiting the authors' look.
- [Yohaï, 2026-10-01] `#history`: review thoroughly the numbers on "The wide gap between passive and active computer
  vision", where some active models appeared above the passive lines. Reviewed 2026-10-01
  (`sources/history-review.md`): no value is wrong; on ImageNet-1k, GFNet (79.8) and Saccader (75.03) sit above the
  dashed line, which holds only self-supervised backbones read out frozen; no active model is above the line trained
  end to end, on either chart. Fixed: the dashed line is labeled "frozen self-supervised" and the notes no longer say
  "below, every time" of it. For the authors: draw the active models against the end-to-end line only (the dashed line
  kept for `#results`, where CanViT-B's frozen probe is like for like); the ImageNet-1k end-to-end line mixes in JFT,
  ImageNet-21k and large teachers from November 2019 on (keep it, saying so, or restrict it to ImageNet-1k);
  Saccader-NASNet (124.5 M) and AME (a ViT-L from SETR weights already trained on ADE20K) are drawn against base-size
  lines; the ADE20K "24.2 points" bracket subtracts AME's 2023 score from DINOv3 ViT-B's 2025 score under DINOv3's own
  protocol (47.19 under the paper's; 10.7 points at AME's date); minor field fixes are listed in the review.
- [Yohaï, 2026-10-01] `#architecture`: the slide looked broken and its top was cut off; visualize the intermediate
  canvases after each read and write, captured as in the paper's canvas evolution figure, with a layout that works.
  Rebuilt 2026-10-01: the first glimpse of the street from the initial canvas, the canvas after each write level with
  it (bundles now record `writeK_canvas` and `initial_canvas`), the next glimpse in place; the drawing 500 px tall,
  checked at 1280 × 720 and in a 1512 × 860 window. Awaiting the authors' look.
- [Yohaï, 2026-10-01] Once the slides are satisfactory, publish them on the gh-pages site; at acceptable checkpoints,
  squash-merge into `main`, `../../deploy.sh --push`, then `../check_published.py`.
- [Yohaï, 2026-10-01] `#cost`: the Adapt column (fine-tuning time, a linear layer) and the Run column (latency per
  glimpse) were rejected, as was the corrected version: cost without comparison points reads as marketing; compare,
  for example, peak performance against training compute. Rebuilt 2026-10-01 as best accuracy against each model's own
  training compute (`OUTLINE.md`); awaiting the authors' look.
- [Yohaï, 2026-10-01] Whether Sabrina gets the slides' data (`site/data/`, gitignored) by a commit on the talk branch, a
  zip, or regeneration; the deployed site (`site/deploy.sh`) now carries it at https://m2b3.github.io/CanViT/ once
  deployed.
- [Yohaï, 2026-10-01] `#glimpses` ("Scenes, viewpoints and glimpses") is unnecessary; instead, one slide introduces the
  concepts and then how the model works, building up gradually with animations (scene, then a glimpse sampled at a
  viewpoint, then what the model does with it), starting from "CanViT in action". Built 2026-10-01: `#rollout`
  ("CanViT in action") moved to its place and staged part by part (`<canvit-episode stage>`), its segmentation's
  largest classes named beside the canvas; awaiting the authors' look.
- [Yohaï, 2026-10-01] `#timeline`: remove subtitles such as "Recurrent glimpses on digits" (a glimpse is not
  recurrent), and establish precisely what was new about each work. Subtitles removed. Each model's novelty, against
  the works before it, read in its paper, and the notes checked sentence by sentence (`sources/active-vision-novelty.md`,
  2026-10-01); the notes corrected from it (AdaptiveNN continues GFNet; Saccader's 75% needs its NASNet classifier; "of
  these seven"). Open: the authors' read of that file.
- [Yohaï, 2026-10-01] `#history`: have Codex investigate missing data; improve the labelling. Labels redone 2026-10-01
  (no key; each passive line named at its end under one "Passive" heading, "linear decoding"; active models named at
  their points over "Active models"); awaiting the authors' look. Codex (gpt-6-astra) audited the data against primary
  sources in two passes (2026-10-01; reports in `throwaway/history-audit/`, merged into `sources/sota-history.json`
  and `.md`): fields filled from official code, the leads resolved, and missing base-size records added. What moved on
  the charts: the ADE20K end-to-end line now starts with Wider or Deeper Model A2 (43.73, 2016) and steps at APCNet
  (45.38, 2019); the frozen ImageNet-1k line starts in 2016 below the axis and enters it in 2019, and Proteus ViT-B
  (84.9, 2024) raises its plateau. Open, in `sota-history.md` ("Unverified leads"): DINOv3's TMLR supplement (ViT-B
  85.3 and ConvNeXt-B 85.4 on ImageNet-1k, ViT-B 52.7 on ADE20K, date unknown); whether the frozen line admits
  supervised distillation (RADIO-AMP-B, 51.2 on ADE20K); SAFER-AiD, a sequential active model with no clean
  full-validation number.
- [Yohaï, 2026-10-01] `#table`: explain what the colors mean, possibly as a probability map of the table class, since
  semantic segmentation is not introduced yet. Done 2026-10-01: DINOv3's probability of table per glimpse, with a
  0-to-1 scale; `#extrapolation` uses the same panel and names its segmentation classes. Awaiting the authors' look.
- [Yohaï, 2026-10-01] `#distillation` needed a complete rework: rebuilt the same day as a training step unrolled in
  time (`OUTLINE.md`); awaiting the authors' look. The predictions shown are the released model's; placeholders
  instead would be a one-line change.
- [Yohaï, 2026-10-01] Show the impact of the viewing policies on ImageNet-1k and ADE20K, with glimpse count on the x
  axis (possibly log-scaled) and top-1 accuracy or mIoU on the y axis; the message is how the policies compare across
  tasks, and that they matter much more for ADE20K. Built as `#policy-accuracy` (after `#results`); awaiting the
  authors' look. What the data says: the early gap is large on both tasks (one glimpse: C2F 76.8 against F2C 32.2
  top-1, 39.6 against 11.0 mIoU); after 21 glimpses the ImageNet-1k policies end within 1.3 points of each other (RFS
  aside), the ADE20K ones 3 points apart and still rising. "Matter much more for ADE20K" holds for the end of the
  rollouts and for C2F against the random policies (the paper's framing), not for the first glimpses.
- [Yohaï, 2026-10-01] Decide what is introduced or highlighted when, as each concept first appears on screen or is
  spoken. Done on the notes (2026-10-01): each term is said where it first occurs: ImageNet, passive vision
  (`#tradition`); glimpse, DINOv3 announced (`#table`); ADE20K (`#history`, named on `#foundation` before it); the
  Vision Transformer, the canvas grids (`#architecture`); R-IID and F-IID (`#distillation`'s notes), the other
  policies (`#policies`); fine-tuning (`#results`); FLOPs (`#cost`). Open: a pass over what each slide shows on screen
  before the speaker names it.
- [Yohaï, 2026-10-01] `<canvit-live>` in the deck failed with a WebAssembly out-of-memory error. Measured on 2026-10-01
  in headless Chromium (`throwaway/live_memory.py`, `slide_memory.py`, `live_wasm.py`): the deck with no slide 376 MB
  RSS; every slide without the model 1.46 GB (static slides about +100 MB each alone; `#rollout` +200, `#memory`
  +210, `#policies` +320); the model created at page open (`autoload`) brought the deck to about 3.5 GB. Fixed: the
  deck now adds `autoload` only when `#live` is the current or next slide (`data-load-near`, `../deck.js`), never in
  the speaker view's previews; with WebGPU disabled the model then starts on WebAssembly in the deck (peak about
  1.9 GB). Open: whether Yohaï's browser (which one, speaker view open?) still fails; releasing the model when the
  slide is left; making the rollouts load only near their slide.
- [Yohaï, 2026-10-01] Give a sense of how good DINOv3 is and how much went into it. Built 2026-10-01 as the last state
  of `#foundation`: 1.7 billion training images, the 6.7-billion-parameter teacher the ViT-B is distilled from,
  61,440 H100-hours to train it (arXiv:2508.10104; `sources/checked-facts.md`); where it stands on dense benchmarks
  is in the notes. Awaiting the authors' look.
- [Yohaï, 2026-10-01] `#foundation`: the DINOv3 feature map was blurry (interpolated instead of nearest) and washed
  out; use other principal components and a higher resolution so objects stand out; possibly shrink it and draw arrows
  to probability maps from the ADE20K segmentation heads, showing that what is where decodes easily; try side by
  side, one after the other, and overlays, since overlays can confuse. In progress: 1024 px input, PCs 2–4 clipped
  2–98%, nearest neighbor (`experiments/foundation`); the released DINOv3 ViT-B ADE20K probe decodes television,
  fireplace, armchair.
- [Yohaï, 2026-10-01] Contrast examples of CanViT with DINOv3 on the full scene at 128 px: beside CanViT's maps,
  DINOv3 ViT-B on the whole scene at 128 px (the same input budget as a glimpse; its released 128 px ADE20K probe,
  `hub.repos.released_dinov3_ade20k_probe("dv3b", input_size_px=128)`), on the rollout, extrapolation and detail
  slides. Done on `#detail` and `#rollout` (each on its last click; `#rollout`'s follows the scene tabs,
  `experiments/whole_scene_baseline`: pixel accuracy 72, 70, 41% against CanViT's 88, 78, 71% after 21 glimpses).
  Not on `#extrapolation`: DINOv3 at 128 px sees the whole table, while the slide shows inference from its two ends;
  the authors' call. The paper's numbers for the matched comparison (passive_comparison_rows: DINOv3 ViT-B at 128 px
  28.8 mIoU against CanViT-B's 29.3 to 39.6 from one full-scene glimpse) say why it matters.
- [Yohaï, 2026-10-01] `#uncertainty` needs a better title; propose options. Proposed "Uncertainty-based viewpoint
  selection" (on the slide), "Entropy-Guided Coarse-to-Fine", "Guiding viewpoint selection with the canvas"; the
  authors choose.
- [Yohaï, 2026-10-01] `#frontier` built step by step as asked (worst policy at 32², at 64², prior models, best
  policy at both) with the axis gliding; awaiting the authors' look. The case for why cost matters, before it, is
  still open (below).
- [Yohaï, 2026-10-01, to keep in mind] Reserve a slide for end-to-end policy learning (reinforcement learning): having
  the observer makes a good policy easy; it will need work, from the unpublished learned-policy work and possibly new
  experiments before the talk. Also teasers of what's next. Both in Backup until the material exists (`OUTLINE.md`).
- [Yohaï, 2026-10-01] The results need a breakdown: the frontier plot covers segmentation only while the talk speaks
  generally; ImageNet-1k and top accuracy need their place. Proposal (to settle with the authors): top accuracy on
  both benchmarks (`#results`: ADE20K 45.9 frozen; ImageNet-1k 84.5 fine-tuned, 81.1 frozen), then the segmentation
  frontier (cost, segmentation only), then ImageNet-1k by glimpse and policy (the paper's Figure 3C; its export to copy
  into `../../assets/paper/` with `copy_paper_figures.sh`): frozen against fine-tuned, how fast accuracy plateaus, C2F
  against F2C again. No efficiency claim on classification.
- [Yohaï, 2026-10-01] The memory slide: retitled; rebuilt on a street (sign, people, bicycle) from a sweep of 6,218
  three-object sequences; awaiting the authors' look (alternative: a bedroom).
- [Yohaï, 2026-10-01] "Spatial coverage and perception of detail" (`#detail`): the idea is good but its presentation
  was not. Rebuilt 2026-10-01 (one example, a zoom into the model's own input), moved into the main talk, its map
  changing on its input's click; awaiting the authors' look.
- [Yohaï, 2026-10-01] `#uncertainty`: "metacognition" said with its scope (`sources/concepts.md`: Fleming 2024 keeps
  "sensitivity to uncertainty" apart from metacognition; Renninger, Verghese & Coughlan 2007: people fixate where
  uncertainty is highest, EG-C2F's rule); decide the wording with the authors.
- [Yohaï, 2026-10-01] Careful, step-by-step, visually supported storytelling on every slide: introduce what a visual
  needs before showing it (`../AGENTS.md`, "Writing slides"). Review every slide for it.
- [Yohaï, 2026-10-01] Reuse the well-received primitives of an earlier deck, ported (never imported): see "Components
  and deck primitives".
- [Yohaï, 2026-10-01] Keep every TODO, known issue and request in this file.
- [Yohaï, 2026-10-01] `#policy-agnosticism`: the slide online was missing data, and is unclear and redundant. The
  missing data: its four photographs, which the deck did not name, so `deploy.sh` did not ship them (fixed: the
  element now names its directory). The slide moved to Backup; its facts are in the notes of `#distillation` and
  `#policies`.
- [Yohaï, 2026-10-01] `#memory`: its role in the story is unclear; Backup, to be reevaluated. Moved to Backup, to be
  reevaluated with the authors: it carries "trans-saccadic integration", and `#neuro-ai`'s
  memory card reuses its maps.
- [Yohaï, 2026-10-01] The deck's data came from gitignored `throwaway/` scripts: read them and commit them alongside
  the website first; refactoring into the core package only where it would make the package better, without ad hoc
  additions to its API; `record_talk.sh` was misnamed (the scripts run experiments; bundles are recordings). The easy
  version is done (2026-10-01): every experiment behind the deck is committed in this talk's `experiments/` and
  reproduces the deployed data exactly; `experiments/build_deck_data.sh` rebuilds `../../data/talk/`; the core package
  is untouched. Open: whether any of it moves into the core package (the authors' call).

## Next

- An independent cold read of the deck (2026-10-01), its precision fixes applied. Open:
  - The neuroscience case is not closed on screen: `#neuro-ai` could call back the Brain-Score figure of
    `#tradition`.
  - "Metacognition" on `#uncertainty` was asserted; now measured (AUROC 0.84, 99% to 37% accuracy from the surest
    tenth of cells to the least sure), said on `#uncertainty` and drawn in Backup (`#calibration`). Whether the number
    belongs on `#uncertainty`'s screen is the authors' call.
  - `#cost` argues cheap adaptation but plots pretraining; the adaptation (0.16 EFLOPs for the ADE20K probe, 4.76 for
    ImageNet-1k LP-FT) is only in the notes.
  - `#neuro-ai`'s top-down feedback card (C2F against F2C) also changes the order of writes; the paper's no-reads
    ablation is a more direct measure of feedback.
  - Pacing: CanViT first appears about ten minutes in, and four benchmark and cost slides run back to back before
    `#neuro-ai`, where neuroscientists may drift.
- [Yohaï, 2026-10-01] `#history` and `#results` fail at their jobs: overloaded, hard to parse, not showing the right
  things; make end-to-end training against frozen features with a linear readout explicit. ImageNet-1k and ADE20K one
  after the other, never side by side from the start, after explaining classification against segmentation and why
  segmentation is harder. Built: the named-state slide with its examples, the passive
  lines restricted to base-size models (never compare against far larger models); `#results` still to rework. M.
- [Yohaï, 2026-10-01] An exploration page for the history data: `_explore-history.html` (a working page, not deployed)
  shows every point with series, size and numeric axes; extend it with training and inference cost once those fields
  exist. S.
- [Yohaï, 2026-10-01] The case for why anyone cares about cost, before the cost charts: money, cheap experiments
 ; training cost first, the main concern for scientists, then inference. `#cost` (best
  accuracy against each model's own training compute, `OUTLINE.md`) and `#frontier` now carry the comparison; the case
  itself is not on screen. Facts that make it concrete: training (pretrained once and released; adapting is a linear
  probe or LP-FT; its pretraining comparable to AdaGlimpse's own, never cheaper than every prior model; `sources/checked-facts.md`), inference GFLOPs per glimpse and per rollout against AME and AdaGlimpse
  (`../../assets/paper/ade20k_seg.json`), wall time per glimpse on a laptop and in the browser. M.
- The cognition links (`sources/cognition.md`, `sources/concepts.md`, read 2026-10-01) are keywords in the notes:
  trans-saccadic integration (`#memory`), boundary extension (`#extrapolation`), coarse-to-fine categorization
  (`#policy-accuracy`), uncertainty-driven fixations (`#uncertainty`). Open: a backup slide with their figures
  (`assets/figures/`: Intraub & Richardson, Biederman et al., Torralba et al., Oliva & Torralba), and citations on
  `#table` (its `XXX`). S.
- `#table`: the title wraps to two lines; a shorter written wording would keep one. S.
- From an open review by Codex (gpt-6-astra, 2026-10-02; its report, `throwaway/talk-review-codex/REPORT.md`, is
  untracked; its precision fixes to the notes are applied), for the authors:
  - Story order: a short `#memory` in the main talk after `#detail`, so "trans-saccadic memory" on `#neuro-ai` has a
    demonstrated referent; `#quickstart` before `#neuro-ai`, so the talk ends on the scientific questions, then
    `#links`.
  - Give one closing question an experimental shape: hold the current glimpse fixed, vary what was seen before, compare
    what is decoded. Human scanpaths need a stated mapping from fixations to crop position and scale.
  - `#extrapolation` without the photograph: the audience must recall the table from `#table`; bring back the scene
    with its two boxes.
  - `#quickstart`: walk through less of the imports and checkpoint names; keep the two glimpse calls and their outputs.
  - If metacognition is a central claim, show that the uncertainty is useful (EG-C2F's early lead over C2F on
    `#policy-accuracy`, or a short `#calibration`).
  - "No policy can make up for a poor observer" heard as a general law: at a fixed budget a policy can matter (the notes
    of `#active-vision-model` give the one-glimpse numbers).
- Open from the second cold read (2026-10-01, a fresh subagent; its fixed defects are in git history), for the
  authors:
  - `#distillation`: the "Target" row did not say it is DINOv3 seeing the whole scene (both reviews tripped on it);
    "DINOv3, whole scene" now sits in small type under "Target". Awaiting the authors' look.
  - `#tradition`: Felleman & Van Essen's hierarchy is illegible at its size (a crop of V1 to IT, or more room).
  - `#detail`: the scene keeps a source watermark; in the loupe state, "CanViT's input, 128 px" shows a magnified crop
    of the input.
  - Subtext: `#cost`'s subtitle restates its axes; `#calibration`'s and `#teacher`'s subtitles are conditions lines.
    `#cost`'s line on what is not counted stays on screen: without it the chart misleads.
  - Conventions: viewpoints are (x, y, scale) on `#distillation` and (row, col) in `#quickstart`'s code; components
    count glimpses from t = 0, `#distillation` and the charts from glimpse 1; `#architecture`'s "after write 1, 2, 3"
    are the paper's Write 0, 1, 2.
  - `#architecture`'s title colors "Vision Transformer" in glimpse blue; the drawing paints it purple.
  - `#rollout`: empty bands above and below the card; DINOv3's heading outweighs CanViT's panel; the canvas shows
    classes beyond the five named.
  - `#history`: the ImageNet-1k base-size line counts 144-crop ensembles (BN-Inception, Inception-v3) as base by
    summing members; the two dashed lines probe DINOv3 ViT-B by different protocols (the paper's on ImageNet-1k,
    DINOv3's on ADE20K); the ADE20K axis runs 10 to 70 with no data below 22.7.
  - `#neuro-ai`: EG-C2F at t = 5 (one box and a quadrant grid) beside Yarbus's dense record invites "the policy looks
    nothing like a scanpath".
  - `#results`' notes quote the paper's "by a wide margin" for both benchmarks; on ImageNet-1k it is +2.3 fine-tuned,
    and frozen CanViT-B is below AdaptiveNN.
  - `#title` names the venue twice (the NeurIPS logo and "NeurIPS 2026"). Backup: `#live` gives entropy as "log 150"
    without its base; `#canvas-attention` keeps the paper's panel letter "A".
- Independent reviews of `OUTLINE.md` (a fresh subagent; `codex exec -m gpt-6-astra`), then the authors'. S.

## Known issues

- `#live` (Backup): shrunk with `zoom: .8`.
- Segmentation colors differ between slides: the components (`#rollout`, `#policies`, `#live`) use the standard
  ADE20K palette (`../../js/canvit/ade20k.js`); `#extrapolation` uses canvit-pytorch's random `LABEL_COLORS`
  (`specialize.ade20k.figures`), in which floor and ceiling are two blues close to glimpse blue; `#history` and
  `#quickstart` give Tableau colors by area, so the names written on regions stay legible. One palette for the talk is
  the authors' call.
- Other colors change meaning between slides (second cold read, 2026-10-01): amber is every prior active model on
  `#history`, `#results` and `#cost`, but AME alone on `#timeline`'s legend, and R-IID's orange sits between them; the
  paper's C2F blue and F2C red are close to glimpse blue and canvas red (on `#neuro-ai`, a blue C2F and a red F2C
  line in the blue card beside the red one); CanViT-B is canvas red on `#results` and `#cost`; the prior models are
  amber dots on `#cost` and other shapes and colors on `#frontier`. The authors choose between the paper's policy
  colors and talk-wide meanings.

## Experiments and exports

| Item | Slides | Diff. | Also | Status |
|---|---|---|---|---|
| Table corners: two glimpses at an object's ends, CanViT against DINOv3 per glimpse, over ADE20K validation (`experiments/table_corners`); to graduate into `canvit_pytorch.viz` | `#table`, `#extrapolation` | S | page | done |
| Distillation: DINOv3's features of a scene and CanViT's prediction after each glimpse (`experiments/distillation`) | `#distillation` | S | page, paper | done |
| Looking closer: small objects missed by a full-scene glimpse and found by a zoomed one (`experiments/looking_closer`); to graduate into `canvit_pytorch.viz` | `#detail` | S | page | done |
| Teacher features of the conference room for the browser (similarity to a hovered patch, PCA, probe segmentation) | `#foundation` | M | page | todo |

## Components and deck primitives

| Item | Slides | Diff. | Status |
|---|---|---|---|
| A periphery that fades to nothing on a click, from blurry to unseen (`<foveated-scene>` on `#human-vision` has the blur): the difference between seeing something blurry and seeing nothing | `#rollout` | S | todo |
| `<canvit-features>`: similarity to the hovered patch, PCA colors, the probe's segmentation | `#foundation` | M | todo |
| The architecture built click by click, pulses along read and write arrows (SVG `animateMotion`), real tokens from a bundle | `#architecture` | M | todo |
| `<canvit-live>` laid out for a slide; one model session for every live slide | `#live` | M | todo |

## Logistics

| Item | Diff. | Status |
|---|---|---|
| Offline on the day: the model's files and manifests come from the browser's cache after one load on the presenting machine (started with the Hub unreachable on 2026-10-01); ONNX Runtime Web loads from jsDelivr (`js/canvit/live-model.js`), offline only from the HTTP cache; install it locally as reveal.js is | S–M | todo |

## For the camera-ready (not the talk)

- Remove "an order of magnitude more than previous active models" from the abstract [Yohaï, 2026-10-01]; the project
  page's abstract and the README copy it and follow.

## Decisions for the authors

- `#architecture` is now a drawing of the two streams on the street rollout, built click by click (glimpse and
  Vision Transformer, canvas, writes, reads, next glimpse with the canvas carried); the paper's full figure, with the
  recurrent CLS, VPE and register tokens, moved to Backup (`#architecture-paper`). Confirm, or put the figure back.
- LookWhere (NeurIPS 2025; 83.0 ImageNet-1k, 44.6 ADE20K, patches selected once from a low-resolution view): address
  it in the talk, or only if asked.
- Learned viewing policies (unpublished): show them or not.
- The funders on the last slide: those of the paper's acknowledgments, confirmed.
- Who presents what, if Sabrina co-presents.
- JAX and MLX ports: clean them up for "runs everywhere", or say PyTorch and the browser.
