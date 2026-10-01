# MAIN 2026 talk: outline

The talk slide by slide (`../AGENTS.md`, "Workflow"). The story, Yohaï's wording and the scoped claims are in
`AGENTS.md`; the open work in `PLAN.md`. Each slide (by its title; its id in `index.html` where cited): where the title's wording comes from, what the screen shows
and how it builds, what is said (notes, in Yohaï's voice), and its status: **ready**, **build** (the visual must be
made from data we have), **data** (an export or experiment is missing), **decide** (the authors' call).

Revision: 2026-10-01, rewritten by Claude Code after reading the reconstructed September transcript in full. Not
yet reviewed by the authors.

## Vision, brains and machines

### Title
- **Shows:** the CanViT wordmark; "Toward Active-Vision Foundation Models"; "the Canvas Vision Transformer"; the four
  authors' faces with their names; McGill, Mila and Université Laval logos; NeurIPS 2026.
- **Says:** "This is CanViT, the Canvas Vision Transformer, joint work with Sabrina Du, Audrey Durand and Suresh
  Krishna."
- **Status:** ready.

### Computational models of biological visual processing
- **Title:** the paper's phrase (§1).
- **Shows:** five classic figures, one per click: receptive fields (Hubel & Wiesel 1962), the Neocognitron
  (Fukushima 1980), the hierarchy of visual areas (Felleman & Van Essen 1991), task performance against IT
  predictivity (Yamins et al. 2014), Brain-Score against ImageNet accuracy, with its inset flattening above 70%
  (Schrimpf et al. 2018).
- **Says:** the tradition, from receptive fields to networks optimized for a task that turn out to predict IT; the
  better the network at the task, the better it predicted the brain, up to a point. The networks used today are
  vision foundation models, and like all of these, they see each image whole, all at once.
- **Status:** ready.

### Human vision is fundamentally active
- **Title:** the UNIQUE application.
- **Shows:** Yarbus 1967: an observer's scanpath over a face; the same painting under seven instructions.
- **Says:** "We have sharp central vision and blurry peripheral vision." We shift our gaze several times per second
  toward what matters, and where we look depends on what we are trying to do. It is sequential: strategic planning,
  evidence integrated across fixations in visual working memory, top-down recurrent feedback (the paper's three
  points and citations).
- **Status:** ready. TODO (`PLAN.md`): a foveated scene, the fixation moving.

## Seeing beyond what you sample

### Integrating multiple viewpoints into a coherent understanding
- **Title:** the UNIQUE application ("integrate information from multiple viewpoints into a coherent understanding");
  "Two glimpses of a table" was "dry and bad" [Yohaï, 2026-10-01]. "What about the space in between? No clue." is said.
- **Shows,** click by click, on the conference room (ADE_val_00001271): the photograph; two glimpses at the two ends
  of the table; a passive model's answer for each glimpse alone (DINOv3 ViT-B and its ADE20K probe), pasted where the
  glimpse was taken, nothing elsewhere; then a ring with a question mark on the empty middle. The answer is said.
- **Says:** "Suppose I see one corner of the table, then another corner. The naïve approach is to paste the
  understanding from here and paste the understanding from there. What about the space in between? No clue." You
  know: more table, because you understand objects and you know the world. That knowledge also tells you where you
  do not need to look.
- **Sources:** `sources/cognition.md` (being verified): which cognitive-science work to cite for inference beyond
  what is sampled.
- **Status:** ready, citations pending. TODO: the passive answers flying from each glimpse to the scene map.

### DINOv3 feature maps
- **Title:** names what the slide shows (Yohaï's phrase for it); "We already know what good visual representations
  look like" is said.
- **Shows:** DINOv3's dense features: the similarity of every patch to a marked one (Siméoni et al. 2025, Fig. 3).
- **Says:** a foundation model is trained once on broad data, without labels, and read out by many tasks; DINOv3
  turns every patch into a vector computed in the context of the whole image; similar vectors mark the parts of the
  same object and objects of the same kind. That is the knowledge of the world the table needs, and the same models
  are among today's best models of visual cortex. They are passive: one image, all at once.
- **Status:** ready. TODO: the interactive version on the conference room (hover a patch, see its similarities,
  PCA colors, the probe's segmentation).

## Active computer vision

### The wide gap between passive and active computer vision
- **Title:** the abstract and the UNIQUE application ("narrows the wide gap between passive and active computer
  vision"); "Active computer vision was not nearly as smart as passive computer vision" is said.
- **Shows:** a chart, year against accuracy, ImageNet-1k top-1 and ADE20K mIoU side by side: the passive frontier
  (with the ImageNet-1k-only and frozen-probe lines), then the sequential active models appearing (DRAM, Saccader,
  GFNet, STAM, AdaGlimpse, AdaptiveNN; AME and AdaGlimpse on ADE20K). No CanViT.
- **Says:** active computer vision goes back to the Recurrent Attention Model, 2014. "Even with lots of training,
  lots of data, and lots of inference compute, it was terrible. The whole point was supposed to be efficiency ...
  Instead, there was worse peak accuracy and worse efficiency at every level." The gap is widest where you need to
  understand the whole scene: segmentation.
- **Data:** `sources/sota-history.json`; caveats in `AGENTS.md`.
- **Status:** build.

### What's in an active-vision model?
- **Title:** the project page's section.
- **Shows:** the page's diagram: Seeing (instantaneous vision, memory: the observer) and Choosing where to look (action
  selection: the policy); then where prior work focused, the policy; then the page's line "No policy can make up for a
  poor observer. A general-purpose observer lets you use any policy."
- **Says:** "I made the same mistake as everybody else: I focused on choosing where to look." "What's an observer?
  It's the thing that makes sense of the world given some inputs ... not just instantaneous vision ... but also
  memory." There was no good observer; borrowing a passive model's weights gave the wrong geometry. "A
  general-purpose observer lets you use any policy."
- **Status:** ready.

## CanViT

### Scenes, viewpoints and glimpses
- **Title:** the paper's §3 heading.
- **Shows:** a scene and the glimpse taken at a viewpoint (position and zoom), the model's 128 px input beside it;
  then the looking-closer example: an object invisible in the zoomed-out glimpse, found by a zoomed-in one.
- **Says:** zooming out trades detail for coverage. Outside the glimpse there is nothing, not even a blur: "a world
  of difference between seeing something blurry and seeing nothing at all".
- **Data:** `throwaway/looking_closer` (running).
- **Status:** ready, example pending.

### Spatial coverage and perception of detail
- **Title:** the paper's §3 ("s_t smoothly controls the tradeoff between spatial coverage and perception of detail").
- **Shows:** two small objects (a clock, a television; ADE_val_00000068, ADE_val_00001195): the scene with the zoomed
  glimpse's box; the zoomed-out glimpse's few pixels of the object; CanViT's probability of its class after the
  zoomed-out glimpse (near zero); on a click, the zoomed-in glimpse; on the next, the probability after it (high).
  Each panel right of the scene shows the box's region.
- **Says:** over 4,151 small objects of ADE20K validation, recall of the object's pixels goes from 18% after the
  full-scene glimpse to 32% after a zoomed one, against 19% for the full scene seen twice: the zoom, not the extra
  step. The two examples are picked; the object is a few pixels when zoomed out, not invisible to a person.
- **Data:** `throwaway/looking_closer` (`summary.log`; panels in `../../data/talk/looking-closer/`).
- **Status:** draft.

### The Canvas Vision Transformer
- **Title:** the paper's.
- **Shows:** the paper's architecture figure, large.
- **Says:** a Vision Transformer backbone sees each glimpse; the canvas holds the scene. "I put the intelligence on the
  Vision Transformer side." The canvas answers, "at any point in time and for any position: what does the model think
  is there?" Reads condition the backbone on the canvas: top-down feedback. Blue for the glimpse, red for the canvas.
- **Status:** ready.

### Canvas Attention
- **Title:** the paper's; "dumb, but not too dumb" is said.
- **Takeaway:** the asymmetry: every learned projection is on the glimpse side; the canvas side has only LayerNorm and
  RoPE.
- **Shows:** a Canvas Attention read–write pair (the paper's Fig. 3A), as large as the slide allows; a click rings the
  four learned projections (glimpse side); the next marks the canvas side; then the line "Learned projections on the
  glimpse side only".
- **Says:** "dumb, but not too dumb": "Just dumb enough that I could interact with it cheaply, while leaving it
  flexible enough to reconstruct unseen things from world-knowledge priors, like the table example." Prior dense
  models had "a cheap encoder and a massive decoder". The cost (2.8 against 37.3 GFLOPs per read–write pair on a
  64 × 64 canvas) is said, not shown.
- **Status:** draft (ring positions to check).

### Passive-to-active dense latent distillation
- **Title:** the paper's.
- **Shows:** the conference room; DINOv3's feature map of the whole scene (PCA colors), the target; CanViT's best
  guess of that map after each of eight random glimpses, in the same colors, filling in.
- **Says:** "We already know what good visual representations look like." Separate the architecture from learning
  the representation: a passive teacher sees the whole scene; the student only gets glimpses, at random places and
  zooms, and "wherever the model was looking, and at whatever zoom, it should be able to produce its best guess about
  the entire scene". Pixel reconstruction would be distillation too, from the identity teacher; this teacher's space
  encodes meaning. No labels anywhere, and no policy built in. The pretraining facts are said here: 13.2 M ImageNet-21k
  scenes, 1 B random glimpses, 166 hours on one H100. A slide of these numbers alone was cut [Yohaï, 2026-10-01:
  "impossible to compare to anything, what is the takeaway"]; it returns only with a verified point of comparison.
- **Data:** `throwaway/distillation` (exported).
- **Status:** build.

## Evidence

### CanViT in action
- **Title:** the project page's "See it in action", with the model named [Yohaï, 2026-10-01].
- **Shows:** `<canvit-episode>` on recorded scenes: the glimpse, the canvas decoded into a segmentation after every
  glimpse.
- **Says:** "It takes glimpses and gradually paints its understanding."
- **Status:** ready.

### Resetting the memory
- **Shows:** `<canvit-path>`: the same smooth path of viewpoints, canvas carried and canvas reset at every glimpse.
- **Says:** reset the memory, and "unless you get a good look at something, the representation is bad".
- **Status:** ready (layout to fit).

### Extrapolation to unobserved regions
- **Title:** the paper's Fig. 1 caption.
- **Shows:** the conference room again: the passive answers per glimpse; CanViT after glimpse A, after B, after A
  then B; segmentation, then the probability of "table".
- **Says:** from the far end alone, the canvas already runs the table toward the near end; with both ends, the table
  is continuous though the middle was never seen. Give the average with its spill (`AGENTS.md`).
- **Status:** ready.

### Viewing policies
- **Shows:** the same scene under each of the paper's policies, glimpse boxes appearing one by one: R-IID and F-IID
  (random viewpoints, as in pretraining; F-IID starts from the full scene), C2F (a quadtree, coarse to fine, random
  order within a level), F2C (the same viewpoints, reversed), EG-C2F (C2F, visiting the most uncertain tile first),
  RFS (the full scene, repeated).
- **Says:** why each exists: R-IID and F-IID are the training distribution; C2F is the natural order; F2C isolates the
  order (same viewpoints by the end, worse result: what was seen changes how the next view is processed); EG-C2F
  shows the canvas can choose where to look without reinforcement learning; RFS separates more processing from new
  input (it helps, then declines).
- **Data:** recorded bundles of one scene per policy (`canvit_pytorch.viz`, same seed).
- **Status:** draft (built; race against the paper's ADE20K curves).

### Benchmark results on ADE20K and ImageNet-1k
- **Title:** the paper's Fig. 3 caption.
- **Shows:** the history chart of `#history`, now with CanViT-B: 45.9% ADE20K mIoU (frozen, linear probe, C2F, 64²
  canvas) and 84.5% ImageNet-1k top-1 (fine-tuned).
- **Status:** ready.

### A new active-vision state of the art in accuracy and efficiency
- **Title:** the project page's section.
- **Shows:** `<canvit-frontier>`: 38.5% from one low-resolution glimpse of the full scene; even fine-to-coarse beats
  the prior models.
- **Says:** "Even our worst policy, a policy worse than random ... got better performance at lower cost."
- **Status:** ready.

## Closing

### Toward action-aware neuro-AI research on vision
- **Title:** the UNIQUE application.
- **Shows:** three questions an active observer lets us ask: what is kept across views; how what was seen changes
  how the next view is processed; how human scanpaths compare with policies on the same observer.
- **Says:** limitations in passing (static scenes, a passive teacher); related work (Thorat et al. 2025, FOVI); "the
  first of many Active-Vision Foundation Models".
- **Status:** ready (visual to find).

### The last slide: paper, code, models and funding
- **Shows:** three QR codes, largest, labeled in a word: "Paper" (arXiv), "Code" (GitHub), "Models" (Hugging
  Face); under them "Homepage: m2b3.github.io/CanViT"; `uv add canvit-pytorch`; the funders' logos in a row. It stays
  up during questions.
- **Status:** ready.

## Backup (place in the story to be settled)

### A general-purpose observer lets you use any policy
- **Title:** the project page's line; a bonus, in Backup [Yohaï, 2026-10-01: "no one [cares] enough for this
  to be the title of a main-talk slide, bonus at best. they care about science and what they can do with it"].
- **Shows:** `<canvit-live>`: I choose where to look; then the canvas's own uncertainty chooses.
- **Status:** ready (layout to fit; offline runtime in `PLAN.md`).

