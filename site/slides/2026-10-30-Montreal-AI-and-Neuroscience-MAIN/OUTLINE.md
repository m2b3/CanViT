# MAIN 2026 talk: outline

The talk slide by slide (`../AGENTS.md`, "Workflow"). The facts and the scoped claims are in
`sources/checked-facts.md`; the open work in `PLAN.md`. Each slide (by its title; its id in `index.html` where cited): where the title's wording comes from, what the screen shows
and how it builds, what is said, and its status: **ready**, **build** (the visual must be
made from data we have), **data** (an export or experiment is missing), **decide** (the authors' call). **Must**
lists what the slide has to show, say or carry across; **Could** what would be interesting or good to show as well
(`../AGENTS.md`, "Workflow"); entries not yet split are being converted.

Revision: 2026-10-01, rewritten by Claude Code; quotes cleaned up 2026-10-23. Not yet reviewed by the authors.

## Vision, brains and machines

### Title
- **Shows:** the CanViT wordmark; "Toward Active-Vision Foundation Models"; the four authors' faces with their names; McGill, Mila and Université Laval logos; NeurIPS 2026.
- **Says:** CanViT, the Canvas Vision Transformer; joint work with Sabrina Du, Audrey Durand and Suresh Krishna.
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
- **Shows:** first the street of the later slides, sharp only around a fixation that jumps from point to point,
  blurrier with eccentricity (`<foveated-scene>`, `foveate.js`; the fixations chosen by hand, an illustration); then
  Yarbus 1967 in its place: an observer's scanpath over a face; the same painting under seven instructions.
- **Says:** central vision is sharp, peripheral vision blurry. We shift our gaze several times per second
  toward what matters, and where we look depends on what we are trying to do. It is sequential: strategic planning,
  evidence integrated across fixations in visual working memory, top-down recurrent feedback (the paper's three
  points and citations).
- **Status:** ready.

## Knowledge of the world

### Integrating multiple viewpoints into a coherent understanding
- **Title:** the UNIQUE application ("integrate information from multiple viewpoints into a coherent understanding");
  "Two glimpses of a table" was rejected as a title [Yohaï, 2026-10-01].
- **Shows,** click by click, on the conference room (ADE_val_00001271): the photograph; two glimpses at the two ends
  of the table; a passive model's answer for the far glimpse alone (DINOv3 ViT-B and its ADE20K probe), appearing over
  its glimpse and flying to its place in an empty scene map; the same for the near glimpse; then a ring with a question
  mark on the empty middle. The answer is said.
- **Says:** pasting each view's passive answer in place leaves the space in between unknown. A person infers more
  table there, from understanding objects and knowing the world.
- **Sources:** `sources/cognition.md` (being verified): which cognitive-science work to cite for inference beyond
  what is sampled.
- **Status:** ready, citations pending.

### DINOv3 feature maps
- **Title:** names what the slide shows (Yohaï's phrase for it).
- **Shows:** a living room; DINOv3 ViT-B's patch features as colors (principal components 2 to 4, nearest neighbor);
  then the features small, with arrows to the probabilities of television, fireplace and armchair that a linear
  probe decodes from them; last, how much went into DINOv3: 1.7 billion training images, no labels; 6.7 billion
  parameters in the 7B model it was distilled from; 61,440 H100-hours to train the 7B model (arXiv:2508.10104:
  §3.1, Fig. 16a, Table 20).
- **Says:** good visual representations already exist. DINOv3 turns every patch into a vector computed in the context
  of the whole image; what is where is easy to read out of it with a linear layer. That is the knowledge of the world
  the table needs, and it comes from scale. These models are passive: one image, all at once.
- **Status:** ready. TODO: the interactive version on the conference room (hover a patch, see its similarities,
  PCA colors, the probe's segmentation).

## Active computer vision

### A brief history of deep active computer vision
- **Title:** Yohaï's [2026-10-01]; it replaced the paper's related-work heading ("Deep active vision").
- **Must:** what active computer vision tried, from RAM (2014) to AdaptiveNN (2025), each model in a few words; that
  almost all learned where to look by reinforcement learning (the paper: prior work "has often focused on action
  selection").
- **Could:** the precursors (Larochelle & Hinton 2010; Bajcsy 1988 "We do not just see, we look") said; the empty
  years before 2019 (digits only).
- **Builds:** a year axis; the models arrive one per click at their release dates (name, citation; no descriptive
  subtitles [Yohaï, 2026-10-01]); then the markers of those whose policy is learned by reinforcement learning turn
  policy teal. Last click [Yohaï, 2026-10-01]: the timeline extends to highlight CanViT, NeurIPS 2026; the axis keeps
  its width, the years squeeze to make room for 2026, every model slides to its new place, and CanViT arrives in the
  brand's gradient (arXiv v1, 23 March 2026; NeurIPS 2026).
- **Data:** `sources/active-vision-timeline.json` (evidence in `.md`).
- **Status:** draft (built).

### The wide gap between passive and active computer vision
- **Title:** the abstract and the UNIQUE application ("narrows the wide gap between passive and active computer
  vision").
- **Must:** what classification and segmentation ask, and why segmentation is harder for a model that sees parts of
  the scene; on each benchmark, active models below passive models of the same size (never against far larger
  ones); end to end against frozen features read by a linear layer, so CanViT's frozen results later read right.
- **Could:** how much training the active models got (glimpses, compute; `sources/checked-facts.md`), backing that
  they lagged despite substantial training; a training-cost view; the all-sizes frontier and the select-once models
  (LookWhere) for questions.
- **Builds** one benchmark after the other, never both at once [Yohaï, 2026-10-01], each introduced by its task
  before its chart: the classification example, large (an ImageNet-1k photo and its one label); it shrinks to the
  chart's corner as the ImageNet-1k chart (year against top-1) draws the passive models trained end to end; then the
  frozen features with a linear layer; then the sequential active models, named. Then the segmentation example, large
  (an ADE20K scene and its labeled map, a few class names written on their regions); it shrinks as the ADE20K chart
  (year against mIoU) draws the same three series; then the bracket of the gap between the best active model and the
  frozen line. No CanViT (`#results`).
- **Says:** the key differences between the two tasks, and why segmentation is harder [Yohaï, 2026-10-01]:
  classification gives one label to the whole image, one of 1000, and ImageNet photos mostly show one object; one good
  look at the right part can be enough. Segmentation gives a label to every pixel of a scene, one of 150 classes,
  large and small things, wall and floor included: you have to know what is everywhere, including where you did not
  look closely, so an active model needs a memory of the whole scene. Most active models cannot produce it at all
  (§1). The score, mIoU: for each class, the overlap between the predicted and true regions over their union,
  averaged over classes. Then the history: active computer vision goes back to the Recurrent Attention Model, 2014;
  it has struggled to match passive models in accuracy and in efficiency, though efficiency was its promise. The gap
  is widest on segmentation.
- **Examples** (chosen by sweeps, `PLAN.md`). Classification: a photo the audience names at a glance, one dominant
  object, typical of ImageNet-1k; its label in plain words and as fine as ImageNet's (a dog's breed shows that the
  1000 classes are fine-grained); a passive classifier confident and right on it. Segmentation: a scene everyone
  recognizes, with many labeled classes of all sizes (large regions such as wall and floor, several small objects),
  nearly no unlabeled pixels, and class names that fit on their regions; not the conference room, which the audience
  will see under other claims.
- **Data:** `sources/sota-history.json`; caveats in `sources/checked-facts.md`.
- **Status:** draft (built).

### What's in an active-vision model?
- **Title:** the project page's section.
- **Shows:** the page's diagram: Seeing (instantaneous vision, memory: the observer) and Choosing where to look (action
  selection: the policy); then where prior work focused, the policy; then what that bought, in two of those papers'
  own ablations (`sources/policy-ablations.json`): random viewpoints against the paper's choice, GFNet on ImageNet-1k
  74.5 against 75.9%, AME on SUN360 72.4 against 73.4% (demonstrate, then name); then, in their place, the page's line
  "No policy can make up for a poor observer. A general-purpose observer lets you use any policy."
- **Says:** prior work, ours included at first, focused on choosing where to look. The observer makes sense of the
  scene from its inputs: instantaneous vision and memory. There was no good observer; borrowing a passive model's
  weights gave the wrong geometry. A general-purpose observer lets you use any policy.
- **Status:** ready.

## CanViT

### CanViT in action
- **Title:** the project page's "See it in action", with the model named [Yohaï, 2026-10-01].
- **Replaces** "Scenes, viewpoints and glimpses", judged unnecessary [Yohaï, 2026-10-01]: this slide introduces the
  concepts and then what the model does, starting from "CanViT in action" and building up gradually: the scene, then
  a glimpse sampled at a viewpoint, then what the model does with it.
- **Builds:** `<canvit-episode stage=...>`, each part fading in at its place: the scene; the viewpoint (its box); the
  crop and the glimpse; CanViT; the canvas after the first glimpse, as features (PCA colors), not as a segmentation
  map [Yohaï, 2026-10-01]; then everything, the rollout playing on from the first glimpse with nothing redrawn and no
  flicker between the gradual reveal and the running animation [Yohaï, 2026-10-01]; the canvas read out as
  segmentation, its largest classes named; last, the episode shrinks left and DINOv3 ViT-B's segmentation of the same
  scene, seen whole at 128 px (one glimpse's pixels, 8 × 8 patches), appears beside it with its pixel accuracy, as
  the contrast [Yohaï, 2026-10-01].
- **Says:** a viewpoint is a position and a scale; the glimpse is its crop at 128 px, all the model gets; seeing
  something blurry differs from seeing nothing; CanViT writes what it understood into the canvas, a memory of the
  whole scene, glimpse after glimpse.
- **Status:** draft (built; the segmentation's largest classes named beside the canvas).

### Canvas Vision Transformer architecture
- **Title:** makes clear that this is the architecture [Yohaï, 2026-10-01]; the paper's Figure 2 is "CanViT
  architecture diagram".
- **Shows:** the paper's two streams drawn for the talk (`architecture.js`), depth left to right, on the street of
  `#rollout`, its first glimpse: the scene and the viewpoint, the glimpse cut into its patches, the Vision
  Transformer's twelve blocks in three groups of four, each followed by the glimpse tokens the next write reads,
  aligned with the blocks [Yohaï, 2026-10-01]. Then the canvas before the glimpse, the initial canvas, the same
  everywhere. Then the glimpse runs through on one click, its reads and writes in depth order [Yohaï, 2026-10-01]:
  each read down from the canvas stream between two blocks; each write up from its glimpse tokens into the canvas it
  leaves, above them, as in the paper's canvas-evolution figure (only the intermediate canvases are shown [Yohaï,
  2026-10-01]): the glimpse's own square, then shapes inside it, then structure beyond it. Then the second glimpse in
  place, the canvas carried over, the same run. Captions say only "Scene", "Canvas", "after write k", "Glimpse" with
  its viewpoint, "Vision Transformer"; no setting details such as the canvas grid [Yohaï, 2026-10-01]. The paper's
  full figure is in Backup.
- **Says:** a Vision Transformer backbone sees each glimpse and does the heavy processing; the canvas holds the scene
  and can be asked, at any time and position, what the model thinks is there. Reads condition the backbone on the
  canvas: top-down feedback. Blue for the glimpse, red for the canvas. That the canvas never goes through a learned
  projection, only a LayerNorm, so it is cheap to read and write, is said here, since Canvas Attention is in Backup.
- **Status:** draft (built 2026-10-01; replaces the paper's figure, `PLAN.md` "Decisions for the authors").

### Passive-to-active dense latent distillation
- **Title:** the paper's §5 (without "policy-agnostic", which the next slide carries).
- **What it shows: the training process**, distinct from the results of the trained model [Yohaï, 2026-10-01]: the
  model produces a whole-scene prediction at each timestep, scored with mean squared error against the teacher
  features at each timestep. The predictions drawn are the released CanViT-B's on that glimpse sequence (real
  outputs, not placeholders).
- **Must:** the teacher, DINOv3 ViT-B, frozen, sees the whole scene at 512 px once and gives a feature vector for every
  patch (32 × 32): the target. The student, CanViT, gets only glimpses (128 px, anywhere, any zoom); after every
  glimpse it must output the teacher's features for the whole scene, seen or not. The loss: squared error between its
  guess and the target, at every position and after every glimpse (plus the CLS token). No labels anywhere.
- **Could:** pixels as the identity teacher (the same recipe with g = identity); the targets precomputed once (about
  8 H100-hours for 13.2 M scenes); the target per-position z-scored.
- **Builds** [Yohaï, 2026-10-01, after rejecting a first version with curved arrows]: something relatively static, a
  training step unrolled in time, one column per glimpse (1, 2, 3, then 21 after an ellipsis), columns aligned, no
  curved arrows, the CanViT logo where CanViT appears, the actual glimpses with their (x, y, scale) as input rather
  than the full scene. One row per role: the 128 px glimpse and its (x, y, scale), the CanViT logo, its prediction of
  DINOv3's features for the whole scene, the mean squared error (a number in a box, colored orange to green as it
  falls [Yohaï, 2026-10-01]; row labels "Prediction", "MSE", "Target" [Yohaï, 2026-10-01]; per-patch heatmaps
  `loss-<t>.png` exist if the authors prefer them), the target (DINOv3 on the whole scene) repeated in every column;
  the canvas carried from logo to logo by a straight arrow. The first column; then its target and error; then a
  column per click, its rows in order from input to output.
- **Data:** `experiments/distillation/training_step.py` writes `steps.json` (viewpoints, the pretraining patch loss after
  each glimpse, and the loss of a prediction of the average target), which `distillation.js` puts on the slide.
- **Says:** good visual representations already exist; wherever the model looks, at whatever zoom, it must produce its
  best guess about the entire scene. Separate the architecture from learning the representation.
- **Example:** a bedroom (ADE_val_00000124), 21 random glimpses.
- **Status:** draft (built).

## What it does

### Extrapolation to unobserved regions
- **Title:** the paper's Fig. 1 caption.
- **Shows:** the conference room again: the passive answers per glimpse; CanViT after glimpse A, after B, after A
  then B; segmentation, then the probability of "table".
- **Says:** from the far end alone, the canvas already runs the table toward the near end; with both ends, the table
  is continuous though the middle was never seen. Give the average with its spill (`sources/checked-facts.md`).
- **Status:** ready.

### Spatial coverage and perception of detail
- **Title:** the paper's §3 ("s_t smoothly controls the tradeoff between spatial coverage and perception of detail").
- **Must:** the model always gets 128 × 128 pixels; zoomed out, a small object is a few pixels and the model misses
  it; zoomed in on it, the same budget shows it and the model finds it. A passive model given the whole scene at that
  budget cannot go and look.
- **Could:** the average (said: recall of the object's pixels 0.18 after the full-scene glimpse, 0.32 after a zoomed
  one, 0.19 for the full scene twice, over 4,151 small objects).
- **Example properties** [stated 2026-10-01]: an object everyone names; a few pixels in the full-scene glimpse; the
  model's p(class) near zero after it and high after the zoom, with few false positives; a scene read at a glance.
  Chosen: ADE_val_00001715, a television in a billiard room (p 0.03 → 0.77; `experiments/looking_closer`, ranked over
  all qualifying objects; runners-up: a basket of soaps, ADE_val_00001081; bottles on a counter, ADE_val_00000439).
- **Builds:** the scene; its full-scene glimpse as the model receives it (128 px, hard nearest), p(television)
  following on the same click, dark; the input zooms into the zoomed glimpse's box, showing the few pixels it had of
  the television; the zoomed glimpse replaces it at the same framing and p(television) lights up on the same click.
  Last, DINOv3 ViT-B's p(television) from the whole scene at 128 px with its released 128 px probe: dark (mean 0.004
  over the object; CanViT 0.03 after the full-scene glimpse, 0.77 after the zoom; `experiments/looking_closer/dinov3_whole_scene.py`).
  The model's answer changes with its input, never on a click of its own [Yohaï, 2026-10-01].
- **Status:** draft (built; rebuilt after the authors judged the idea good and the first presentation poor). In the
  main talk, after Extrapolation [Yohaï, 2026-10-01].

### Viewing policies
- **Must:** what each policy does and why it is in the paper, in its paper color; the same CanViT-B works under all
  of them without retraining; C2F against F2C: the same viewpoints in another order give a worse result, so what was
  seen changes how the next view is processed.
- **Must** [Yohaï, 2026-10-01: metacognition, and the model's own uncertainty about what is where, emphasized on the
  policies, particularly EG-C2F]: EG-C2F as metacognition, the model's own uncertainty about what is where, explained
  before it is used: at one position, the class probabilities read out from the canvas as bars, peaked (confident) or
  flat (uncertain); entropy as the number for how flat; the entropy map over the scene; EG-C2F visiting next the tile
  of highest mean entropy; the policy reads the uncertainty, the canvas does not act.
- **Could:** the race over all of ADE20K validation; EG-C2F, whose viewpoints follow the uncertainty of the
  segmentation read out from the canvas, without reinforcement learning; RFS, more processing without new input, helping then declining.
- **Builds** [Yohaï, 2026-10-01: introduce the policies one after the other rather than all moving at once, without
  their ADE20K performance, which other slides show]: one policy per click, R-IID, F-IID, C2F, F2C, EG-C2F, RFS, each a
  recorded rollout of the street played in its color on one scene (the deck switches the bundle); beside it the
  policies so far, label and name, the current one in its color. No accuracy here (`#policy-accuracy`). The earlier
  build was a grid of cards with each policy's color and title, fading into the six rollouts playing together, then
  the race.
- **Says:** why each exists: R-IID and F-IID are the training distribution; C2F is the natural order; F2C isolates the
  order (same viewpoints by the end); EG-C2F looks first where the model is least sure, without reinforcement learning
  (the next slide explains how); RFS separates more processing from new input.
- **Data:** the street's recorded bundles, one per policy (`canvit_pytorch.viz`, seed 0).
- **Status:** draft (rebuilt 2026-10-01).

### Uncertainty-based viewpoint selection
- **Title:** the paper's §6 ("EG-C2F's uncertainty-based viewpoint selection"), proposed 2026-10-01 after "The
  model's own uncertainty about what's where" was rejected as a title [Yohaï, 2026-10-01]; alternatives offered:
  "Entropy-Guided Coarse-to-Fine" (the policy's name), "Guiding viewpoint selection with the canvas" (§6: "the
  ability of the canvas to guide viewpoint selection"). The model's own uncertainty about what is where, and
  metacognition, are said.
- **Must:** metacognition, explained: at a position, the class probabilities read out from the canvas, peaked or
  flat; entropy as how flat; the uncertainty map; EG-C2F averaging it per quadrant and looking next where it is
  highest, without reinforcement learning.
- **Could:** the next glimpses one by one; AME's attention-map entropy as the precedent (said).
- **Example:** the street of `#policies` after its first glimpse (continuity with the race); the surest and least
  sure canvas cells chosen by entropy (`experiments/metacognition`).
- **Status:** draft (built).

## Results

### Benchmark results on ADE20K and ImageNet-1k
- **Title:** the paper's Fig. 3 caption.
- **Shows:** the history chart of `#history`, now with CanViT-B: 45.9% ADE20K mIoU (frozen, linear probe, C2F, 64²
  canvas) and 84.5% ImageNet-1k top-1 (fine-tuned), named in the chart's margin.
- **Builds:** ImageNet-1k with CanViT-B's two results; then ADE20K with its result; then a dotted line at AME's level
  and an arrow up to CanViT-B's 45.9, "+18.3" (the paper: "up from 27.6%"). ImageNet-1k gets no arrow: 2.3 points
  over AdaptiveNN is about 30 px and crowds AdaptiveNN's point; it is in the notes.
- **Status:** draft (the breakdown of results is still to settle, `PLAN.md`).

### Accuracy by viewing policy
- **Title:** names what the charts show (the paper's Figure 3B and 3C: "mIoU by viewing policy", "classification
  accuracy by viewing policy").
- **Request** [Yohaï, 2026-10-01]: show the impact of the viewing policies on ImageNet-1k and on ADE20K, glimpse count
  on the x axis (possibly log-scaled); the message is how the policies compare with one another and across tasks, and
  that they matter much more on ADE20K; axis cutoffs chosen for the story.
- **Must:** on ImageNet-1k the policies start far apart and end together (within about a point, RFS aside); on
  ADE20K they stay apart and keep rising; C2F ends above F2C on both, with the same viewpoints.
- **Builds:** `<canvit-policy-curves>`, glimpses on a log axis; ImageNet-1k first (frozen, 32² canvas, 70 to 82%),
  then ADE20K beside it (frozen, 64² canvas, 35 to 47 mIoU): two windows of the same 12 points, so the spreads
  compare; low starts enter from below.
- **Scope:** "matter much more for ADE20K" holds at the end of the rollouts; after one glimpse the gap is large on both
  (`PLAN.md`).
- **Status:** draft (built).

### Training cost
- **Title:** "Training cost", subtitle "Best accuracy v. each model's own training compute".
- **Why:** the case for cost before the frontier [Yohaï, 2026-10-01]: why anyone should care about efficiency (money,
  cheaper experiments); for scientists, training cost is the main concern, inference cost second. The first version
  (train once, adapt, run: three large numbers) was rejected [Yohaï, 2026-10-01]: latency does not matter to this
  audience, and numbers without comparison points are marketing; compare peak performance against training compute
  instead.
- **Shows:** one chart per benchmark, ImageNet-1k first, then ADE20K beside it: each active model's best accuracy
  against its own training compute (EFLOPs, log), from `experiments/training_cost/export.py`, which applies `accounting.py` and adds CanViT-B's adaptation (ImageNet-1k fine-tuning 4.76 EFLOPs, the ADE20K probe 0.16)
  with the same formulas. ImageNet-1k: CanViT-B 55.3 frozen (81.1) and 60.1 fine-tuned (84.5); AdaGlimpse 62.9 to
  144.6 (77.5; its code and its paper differ on pretraining glimpses); AdaptiveNN 22.6 (82.2), counted from its code
  and paper [Yohaï, 2026-10-01]. ADE20K: CanViT-B 55.5 (45.9); AME 1.4 to 1.8 on top of pretrained ViT-L weights
  (27.6); AdaGlimpse 83.7 to 166.6 over its pretraining, reconstruction and segmentation stages (25.7). The counts:
  `sources/baseline-training-compute.md`. One line under both: the training of the teachers and pretrained
  weights each model starts from is not counted (the accounting's rule; a teacher's inference is, such as CanViT-B's
  DINOv3 features). AME's label names the SETR weights, already trained on ADE20K, that it starts from.
- **Says:** compute is money and time; then the comparison; that the teachers are not counted, DINOv3 included; one
  model for both tasks and any policy.
- **Status:** draft (built 2026-10-01).

### Active ADE20K segmentation
- **Title:** "Active ADE20K segmentation", subtitle "Accuracy v. efficiency" [Yohaï, 2026-10-01]. Before it, the
  paper's Figure 3A caption scoped to what the chart shows: active models only, segmentation only [Yohaï,
  2026-10-01]; "A new active-vision state of the art in accuracy and efficiency" (the page's section) spoke of more
  than segmentation, which this plot alone does not support [Yohaï, 2026-10-01]. No best-prior reference line
  [Yohaï, 2026-10-01].
- **Builds** [Yohaï, 2026-10-01: not everything at once; the worst policy at 32², then at 64², then the prior active
  models, then the best policy at both resolutions]: F2C at 32²; F2C at 64²; the prior active models (the cost axis
  widens to 835 GFLOPs as they arrive, pushing CanViT's curves left); EG-C2F at 32² and 64². `<canvit-frontier
  series=...>` set by fragments; each curve labeled at its end.
- **Shows:** `<canvit-frontier>`: 38.5% from one low-resolution glimpse of the full scene; even fine-to-coarse beats
  the prior models.
- **Must:** why cost matters (money and time per experiment), training cost first, the main concern for scientists
  [Yohaï, 2026-10-01]: what CanViT-B cost to train against the prior active models, each system's own training
  (`sources/checked-facts.md`: hours, EFLOPs, glimpses, with their qualifiers); then inference: from a single glimpse
  of the full scene, CanViT-B beats the best prior active model with 20 times fewer inference FLOPs, and even F2C
  beats it; scoped to segmentation (no efficiency claim on classification).
- **Could:** cost per glimpse and to train against an AdaGlimpse-style design on the same harness (22 against 212
  TPU chip-hours); the model running on a laptop or in a browser.
- **Says:** first why cost matters to this audience [Yohaï, 2026-10-01]: compute is money and time; a model cheaper
  to run makes the experiments that run it cheaper (more scenes, glimpses and policies, policies learned by trial and
  error) and runs on a laptop; the twentyfold is inference FLOPs on segmentation, never a multiplier on an
  experiment's cost. Then: even the worst policy, worse than random, beats the best prior model at lower cost.
- **Status:** draft: the case for cost is not on the slide yet (`PLAN.md`).

## Closing

### Toward action-aware neuro-AI research on vision
- **Title:** the UNIQUE application.
- **Shows:** three questions an active observer lets us ask, each on a card over the demonstration in this talk that
  shows CanViT can be asked it, one card per click, in the colors of the three parts of an active-vision model
  (`#active-vision-model`): top-down feedback (instantaneous vision, blue): how what was seen changes how the next
  view is processed, over C2F against F2C on ADE20K (`<canvit-policy-curves policies>`); trans-saccadic memory
  (memory, red): what is kept from one view to the next, over the street of `#memory` (in Backup) kept and reset; scanpaths
  (action selection, teal): how human scanpaths compare with policies on the same observer, over Yarbus's record
  beside EG-C2F's first glimpses. The questions are CC's synthesis from the paper and the UNIQUE application [CC,
  2026-10-01; for the authors to confirm]. Alternative not drawn: a bookend of `#human-vision`, each property of human
  vision the paper lists (gaze shifts and planning, integration in working memory, top-down feedback) beside its
  CanViT counterpart (policy, canvas, reads).
- **Says:** limitations in passing (static scenes, a passive teacher); related work (Thorat et al. 2025, FOVI); "the
  first of many Active-Vision Foundation Models".
- **Status:** draft (built).

### Quickstart
- **Title:** the README's heading.
- **Must:** that using CanViT takes a few lines anyone can read, so starting to use it is trivial: install, load,
  glimpse, read the canvas [Yohaï, 2026-10-01].
- **Code** [Yohaï, 2026-10-01: no loop, glimpses written inline one after the other, neither defined upfront]:
  `quickstart.py`, the segmentation model, each glimpse written where it is taken; it runs as shown
  (`experiments/quickstart/export.py` executes it). The second viewpoint, the left of the street, was chosen by
  `experiments/quickstart/sweep.py`: it adds person and van.
- **Builds:** `uv add canvit-pytorch` alone and large; the code; a band walks through it: the model, the scene and
  state (the street appears), the first glimpse (its box, then `logits.argmax(1)` with the class names), the second
  (its box, then the new map).
- **Status:** draft.

### The last slide: paper, code, models and funding
- **Shows:** three QR codes, largest, labeled in a word: "Paper" (arXiv), "Code" (GitHub), "Models" (Hugging
  Face); under them "Homepage: m2b3.github.io/CanViT"; `uv add canvit-pytorch`; the funders' logos in a row. It stays
  up during questions.
- **Status:** ready.

## Backup (place in the story to be settled)

A divider slide, "Backup", opens the section [Yohaï, 2026-10-01].

### End-to-end policy learning (reserved)
- **Title:** to choose with the authors ("What about RL?" was floated as a spoken framing).
- **Must:** with a good observer, learning a policy becomes easy: a policy trained on top of the frozen CanViT-B
  (the VPE token exists for this) and how it compares with C2F and EG-C2F [Yohaï, 2026-10-01: this will need work].
- **Data:** the unpublished learned-policy work and any experiment run before the talk; nothing is shown before it is
  read and verified.
- **Status:** reserved.

### What's next (reserved)
- **Must:** teasers of the work that follows, as the authors decide [Yohaï, 2026-10-01].
- **Status:** reserved.

### Policy agnosticism
- **In Backup** [Yohaï, 2026-10-01: unclear and redundant in the main talk]: its facts are said on
  `#distillation` (random viewpoints and lengths, the training scale) and `#policies` (which policies pretraining used).
- **Title:** the paper's §5.2 heading.
- **Must:** during pretraining every viewpoint is random: position anywhere, zoom from the whole scene down to 0.25%
  of it, small glimpses favored (p(s) ∝ 1 − s), sequences of random length (mean 4, sometimes much longer); two
  rollouts per scene, one random from the start (R-IID), one starting from the full scene (F-IID). So no policy is
  built in, and any policy can be used afterwards (C2F, F2C, EG-C2F were never seen in training).
- **Could:** the scale distribution as a histogram; the training scale (13.2 M scenes, about 1 B glimpses, 166 h on
  one H100) said here.
- **Builds:** several scenes, each with its random glimpse boxes appearing one by one, lengths differing, an R-IID and
  an F-IID rollout side by side; then the histogram of zooms; viewpoints drawn by `canvit_pytorch.policies` itself
  and exported, never re-implemented in the page.
- **Status:** draft (built: four ImageNet photographs, R-IID then F-IID, the pretraining sampler's own draws; the
  histogram of zooms not drawn).

### A persistent, evolving understanding of the scene
- **In Backup, to be reevaluated** [Yohaï, 2026-10-01: its role in the story is unclear]. `#neuro-ai`'s
  trans-saccadic memory card shows its kept and reset maps.
- **Title:** the paper's and the page's words for memory ("updating a persistent, evolving understanding of the
  scene"); "Resetting the memory" named a condition, not what the slide shows [Yohaï, 2026-10-01].
- **Must:** trans-saccadic integration made visible: each glimpse adds what it saw to one picture of the scene, and
  what was seen stays; with the memory reset before each glimpse, only the current glimpse remains.
- **Could:** the average over ADE20K validation (said: 0.46 kept against 0.03 reset, single objects).
- **Example properties** [stated 2026-10-01, before a new sweep; the person in a desert (ADE_val_00001425) was clean
  but not inspiring]: a scene the audience relates to and finds lively (a street with people and
  cars, a kitchen, a living room), well photographed; three objects of distinct, nameable classes, each needing its own
  glimpse and recognized confidently after it; glimpses mostly disjoint; after the last glimpse, the kept canvas shows
  all three with readable shapes and the reset canvas only the last.
- **Chosen** (2026-10-01, `experiments/memory`, `sources/checked-facts.md`): ADE_val_00000836, a street: a shop sign,
  people walking, a bicycle; alternative ADE_val_00001182, a bedroom (lamp, towels, painting), cleaner numbers, less
  lively.
- **Builds:** each class's probability in its own color over the dimmed scene, a glimpse per click: the first object
  lights up, then the second while the first stays, then the third; then, beside it, the same glimpses with the memory
  reset: only the last object.
- **Says:** an object is represented well only after a good glimpse of it; the average, as a clean example's
  context.
- **Status:** draft (built).

### Canvas Attention
- **Title:** the paper's.
- **Place:** Backup [Yohaï, 2026-10-01: Canvas Attention may not be relevant to this audience; perhaps backup only]:
  its point for this audience fits in a sentence, said on Canvas Vision Transformer architecture; the read–write pair
  and its projections are machine-learning detail.
- **Takeaway:** the asymmetry: every learned projection is on the glimpse side; the canvas side has only LayerNorm and
  RoPE.
- **Shows:** a Canvas Attention read–write pair (the paper's Fig. 3A), as large as the slide allows; a click rings the
  four learned projections (glimpse side); the next marks the canvas side.
- **Says:** the canvas is cheap to interact with, yet flexible enough to reconstruct unseen things from
  world-knowledge priors, like the table example. Prior dense models paired a small encoder with a large decoder
  (§2: an MAE-style decoder over the full grid at every step). The cost (2.8 against 37.3 GFLOPs per read–write pair
  on a 64 × 64 canvas) is said, not shown.
- **Status:** draft (ring positions to check).

### CanViT architecture diagram
- **Title:** the paper's Figure 2 caption.
- **Shows:** the paper's figure: the two streams over three glimpses with the recurrent CLS token, the VPE token and
  the registers, for questions about what `#architecture` leaves out.
- **Status:** draft, in Backup.

### CanViT-B and its DINOv3 teacher
- **Title:** names what the chart compares; subtitle "ADE20K accuracy v. inference compute". For the question
  "isn't it just DINOv3?".
- **Builds:** `<canvit-frontier series="dinov3">`, log cost axis, mIoU 25 to 50: DINOv3 ViT-B/16 on the whole scene at
  128 to 512 px (the export's probe_table, the paper's Figure 4C), then CanViT-B under EG-C2F at 32² and 64² canvases.
- **Says:** at small budgets CanViT-B is above its own teacher (38.5 from one glimpse against 33.2 for the teacher at the
  same compute, the paper's macros); with the whole scene at 512 px the teacher ends higher (47.2).
- **Status:** draft (built), in Backup.

### Accuracy by uncertainty
- **Title:** names what the chart shows; subtitle "ADE20K validation, every canvas cell".
- **For:** "is that uncertainty worth anything?" after `#uncertainty`, and the metacognition keyword's scope.
- **Shows:** accuracy of the decoded class in each tenth of canvas cells ranked by entropy, after the full-scene glimpse
  and after five C2F glimpses: 99% where surest, 37% (41%) where least sure; AUROC 0.84 in the notes.
- **Status:** draft (built 2026-10-01), in Backup.

### Live demo
- **Title:** "Live demo" [Yohaï, 2026-10-01], replacing the page's line "A general-purpose observer lets you use any
  policy"; a bonus, in Backup, not a main-talk slide [Yohaï, 2026-10-01].
- **Shows:** `<canvit-live>`: I choose where to look; then EG-C2F chooses, from the uncertainty of the segmentation read out from the canvas.
  It runs CanViT-B and its ADE20K probe from the Hub (`canvit/…-c64-onnx-fp32`, `canvit/probe-…-onnx-fp32`, published
  2026-10-01 after parity and browser checks).
- **Status:** ready (layout to fit; offline runtime in `PLAN.md`).
