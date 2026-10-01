# MAIN 2026 talk: outline

The talk slide by slide (`../AGENTS.md`, "Workflow"). The story, Yohaï's wording and the scoped claims are in
`AGENTS.md`; the open work in `PLAN.md`. Each slide (by its title; its id in `index.html` where cited): where the title's wording comes from, what the screen shows
and how it builds, what is said (notes, in Yohaï's voice), and its status: **ready**, **build** (the visual must be
made from data we have), **data** (an export or experiment is missing), **decide** (the authors' call). **Must**
lists what the slide has to show, say or carry across; **Could** what would be interesting or good to show as well
(`../AGENTS.md`, "Workflow"); entries not yet split are being converted.

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

## Knowledge of the world

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

### A brief history of deep active computer vision
- **Title:** Yohaï's [2026-10-01]; it replaced the paper's related-work heading ("Deep active vision").
- **Must:** what active computer vision tried, from RAM (2014) to AdaptiveNN (2025), each model in a few words; that
  almost all learned where to look by reinforcement learning (the paper: prior work "has often focused on action
  selection").
- **Could:** the precursors (Larochelle & Hinton 2010; Bajcsy 1988 "We do not just see, we look") said; the empty
  years before 2019 (digits only).
- **Builds:** a year axis; the models arrive one per click at their release dates (name, citation) [Yohaï, 2026-10-01, on the descriptions: "remove the [bad]
  subtitles"];
  then the markers of those whose policy is learned by reinforcement learning turn policy teal.
- **Data:** `sources/active-vision-timeline.json` (evidence in `.md`).
- **Status:** draft (built).

### The wide gap between passive and active computer vision
- **Title:** the abstract and the UNIQUE application ("narrows the wide gap between passive and active computer
  vision"); "Active computer vision was not nearly as smart as passive computer vision" is said.
- **Must:** what classification and segmentation ask, and why segmentation is harder for a model that sees parts of
  the scene; on each benchmark, active models below passive models of the same size (never against far larger
  ones); end to end against frozen features read by a linear layer, so CanViT's frozen results later read right.
- **Could:** how much training the active models got (glimpses, compute; `AGENTS.md`, rebuttal facts), backing "even
  with lots of training"; a training-cost view; the all-sizes frontier and the select-once models (LookWhere) for
  questions.
- **Builds** one benchmark after the other, never both at once [Yohaï, 2026-10-01], each introduced by its task
  before its chart: the classification example, large (an ImageNet-1k photo and its one label); it shrinks to the
  chart's corner as the ImageNet-1k chart (year against top-1) draws the passive models trained end to end; then the
  frozen features with a linear layer; then the sequential active models, named. Then the segmentation example, large
  (an ADE20K scene and its labeled map, a few class names written on their regions); it shrinks as the ADE20K chart
  (year against mIoU) draws the same three series; then the bracket of the gap between the best active model and the
  frozen line. No CanViT (`#results`).
- **Says:** the two tasks [Yohaï, 2026-10-01: "we must explain the key differences between classification and
  segmentation and why one is harder"]: classification gives one label to the whole image, one of 1000, and ImageNet
  photos mostly show one object; one good look at the right part can be enough. Segmentation gives a label to every
  pixel of a scene, one of 150 classes, large and small things, wall and floor included: you have to know what is
  everywhere, including where you did not look closely, so an active model needs a memory of the whole scene. Most
  active models cannot produce it at all (§1). The score, mIoU: for each class, the overlap between the predicted and
  true regions over their union, averaged over classes. Then the history: active computer vision goes back to the
  Recurrent Attention Model, 2014. "Even with lots of training, lots of data, and lots of inference compute, it was
  terrible. The whole point was supposed to be efficiency ... Instead, there was worse peak accuracy and worse
  efficiency at every level." The gap is widest on segmentation.
- **Examples** (chosen by sweeps, `PLAN.md`). Classification: a photo the audience names at a glance, one dominant
  object, typical of ImageNet-1k; its label in plain words and as fine as ImageNet's (a dog's breed shows that the
  1000 classes are fine-grained); a passive classifier confident and right on it. Segmentation: a scene everyone
  recognizes, with many labeled classes of all sizes (large regions such as wall and floor, several small objects),
  nearly no unlabeled pixels, and class names that fit on their regions; not the conference room, which the audience
  will see under other claims.
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
- **Shows:** a scene and the glimpse taken at a viewpoint (position and zoom), the model's 128 px input beside it.
  The looking-closer example is its own slide (Spatial coverage and perception of detail).
- **Says:** zooming out trades detail for coverage. Outside the glimpse there is nothing, not even a blur: "a world
  of difference between seeing something blurry and seeing nothing at all".
- **Status:** ready.

### Canvas Vision Transformer architecture
- **Title:** says it is the architecture [Yohaï, 2026-10-01: "this slide should be renamed to make it clear this is
  architecture"]; the paper's Figure 2 is "CanViT architecture diagram".
- **Shows:** the paper's architecture figure, large.
- **Says:** a Vision Transformer backbone sees each glimpse; the canvas holds the scene. "I put the intelligence on the
  Vision Transformer side." The canvas answers, "at any point in time and for any position: what does the model think
  is there?" Reads condition the backbone on the canvas: top-down feedback. Blue for the glimpse, red for the canvas.
  The memory "dumb, but not too dumb" (the canvas never goes through a learned layer), cheap to read and write, is said
  here, since Canvas Attention is in Backup.
- **Status:** ready.

### Passive-to-active dense latent distillation
- **Title:** the paper's §5 (without "policy-agnostic", which the next slide carries).
- **What it shows: the training process** [Yohaï, 2026-10-01: "absolutely distinguish the TRAINING PROCESS from the
  RESULT with the trained model"; then: "you basically just want to show that the model produces a whole-scene
  prediction at each timestep and that it is scored with mean squared error against the teacher features at each
  timestep"]. The predictions drawn are the released CanViT-B's on that glimpse sequence (real outputs, not
  placeholders).
- **Must:** the teacher, DINOv3 ViT-B, frozen, sees the whole scene at 512 px once and gives a feature vector for every
  patch (32 × 32): the target. The student, CanViT, gets only glimpses (128 px, anywhere, any zoom); after every
  glimpse it must output the teacher's features for the whole scene, seen or not. The loss: squared error between its
  guess and the target, at every position and after every glimpse (plus the CLS token). No labels anywhere.
- **Could:** pixels as the identity teacher (the same recipe with g = identity); the targets precomputed once (about
  8 H100-hours for 13.2 M scenes); the target per-position z-scored.
- **Builds** [Yohaï, 2026-10-01, on the first version: "completely terrible ... the fancy curved arrows etc do nothing
  for it. having something relatively static where you unroll the timesteps across time ... you can duplicate the
  teacher features, we want things to align in terms of columns etc. this should not need curved arrows"; "when you
  have CanViT in a diagram use the logo"; "show the actual glimpses not the full scene, the input is the glimpse +
  x,y,scale so show THAT"; "go up to glimpse 21"; "for the MSE loss, we might want a heatmap, or a single number"]: a
  training step unrolled in time, one column per glimpse (1, 2, 3, then 21 after an ellipsis), one row per role: the
  128 px glimpse and its (x, y, scale), the CanViT logo, its prediction of DINOv3's features for the whole scene, the
  mean squared error (a number in a box, orange to green as it falls [Yohaï: "COLOR THE MSE TEXT to make it clear it
  gets better"]; row labels "Prediction", "MSE", "Target" [Yohaï]; per-patch heatmaps `loss-<t>.png` exist if the authors prefer them), the target
  (DINOv3 on the whole scene) repeated in every column; the canvas carried from logo to logo by a straight arrow. The
  first column; then its target and error; then a column per click, its rows in order from input to output.
- **Data:** `throwaway/distillation/training_step.py` writes `steps.json` (viewpoints, the pretraining patch loss after
  each glimpse, and the loss of a prediction of the average target), which `distillation.js` puts on the slide.
- **Says:** "We already know what good visual representations look like." "Wherever the model was looking, and at
  whatever zoom, it should be able to produce its best guess about the entire scene." Separate the architecture from
  learning the representation.
- **Example:** the bedroom of the next two slides (ADE_val_00000124).
- **Status:** to build.

### Policy agnosticism
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
- **Status:** to build.

## What it does

### CanViT in action
- **Title:** the project page's "See it in action", with the model named [Yohaï, 2026-10-01].
- **Shows:** `<canvit-episode>` on recorded scenes: the glimpse, the canvas decoded into a segmentation after every
  glimpse.
- **Says:** the canvas read out with a linear layer, as DINOv3 was on DINOv3 feature maps; "It takes glimpses and
  gradually paints its understanding."
- **Status:** ready.

### A persistent, evolving understanding of the scene
- **Title:** the paper's and the page's words for memory ("updating a persistent, evolving understanding of the
  scene"); "Resetting the memory" named a condition, not what the slide shows [Yohaï, 2026-10-01: "why would a slide
  be called that"].
- **Must:** trans-saccadic integration made visible: each glimpse adds what it saw to one picture of the scene, and
  what was seen stays; with the memory reset before each glimpse, only the current glimpse remains.
- **Could:** the average over ADE20K validation (said: 0.46 kept against 0.03 reset, single objects).
- **Example properties** [stated 2026-10-01, before a new sweep; the person in a desert (ADE_val_00001425) was clean
  but "really doesn't look inspiring"]: a scene the audience relates to and finds lively (a street with people and
  cars, a kitchen, a living room), well photographed; three objects of distinct, nameable classes, each needing its own
  glimpse and recognized confidently after it; glimpses mostly disjoint; after the last glimpse, the kept canvas shows
  all three with readable shapes and the reset canvas only the last.
- **Chosen** (2026-10-01, `throwaway/memory_reset/trio_*`, `AGENTS.md`): ADE_val_00000836, a street: a shop sign,
  people walking, a bicycle; alternative ADE_val_00001182, a bedroom (lamp, towels, painting), cleaner numbers, less
  lively.
- **Builds:** each class's probability in its own color over the dimmed scene, a glimpse per click: the first object
  lights up, then the second while the first stays, then the third; then, beside it, the same glimpses with the memory
  reset: only the last object.
- **Says:** "unless you get a good look at something, the representation is bad"; the average, as a clean example's
  context.
- **Status:** draft (built).

### Extrapolation to unobserved regions
- **Title:** the paper's Fig. 1 caption.
- **Shows:** the conference room again: the passive answers per glimpse; CanViT after glimpse A, after B, after A
  then B; segmentation, then the probability of "table".
- **Says:** from the far end alone, the canvas already runs the table toward the near end; with both ends, the table
  is continuous though the middle was never seen. Give the average with its spill (`AGENTS.md`).
- **Status:** ready.

### Spatial coverage and perception of detail
- **Title:** the paper's §3 ("s_t smoothly controls the tradeoff between spatial coverage and perception of detail").
- **Must:** the model always gets 128 × 128 pixels; zoomed out, a small object is a few pixels and the model misses
  it; zoomed in on it, the same budget shows it and the model finds it. A passive model given the whole scene at that
  budget cannot go and look.
- **Could:** the average (said: recall of the object's pixels 0.18 after the full-scene glimpse, 0.32 after a zoomed
  one, 0.19 for the full scene twice, over 4,151 small objects); DINOv3 ViT-B at 128 px on the same scene, beside it
  (`PLAN.md`).
- **Example properties** [stated 2026-10-01]: an object everyone names; a few pixels in the full-scene glimpse; the
  model's p(class) near zero after it and high after the zoom, with few false positives; a scene read at a glance.
  Chosen: ADE_val_00001715, a television in a billiard room (p 0.03 → 0.77; `throwaway/looking_closer`, ranked over
  all qualifying objects; runners-up: a basket of soaps, ADE_val_00001081; bottles on a counter, ADE_val_00000439).
- **Builds:** the scene; its full-scene glimpse as the model receives it (128 px, hard nearest), p(television)
  following on the same click, dark; the input zooms into the zoomed glimpse's box, showing the few pixels it had of
  the television; the zoomed glimpse replaces it at the same framing and p(television) lights up on the same click.
  The model's answer changes with its input, never on a click of its own [Yohaï, 2026-10-01: "please think through
  when things should update and how, i shouldnt need an additional right-arrow to make the probability map
  update"].
- **Status:** draft (built; rebuilt after "the idea ... is good and nice but the way it is showed is really not good
  atm"). In the main talk, after Extrapolation [Yohaï, 2026-10-01: "the 'Spatial coverage and perception of detail'
  stuff is really great and should be in main pres"].

### Viewing policies
- **Must:** what each policy does and why it is in the paper, in its paper color; the same CanViT-B works under all
  of them without retraining; C2F against F2C: the same viewpoints in another order give a worse result, so what was
  seen changes how the next view is processed.
- **Must** [Yohaï, 2026-10-01: "a keyword i really want to see emphasized on the policy stuff, particularly EG-C2F,
  is the idea of metacognition, and the model's own uncertainty about what's where"]: EG-C2F as metacognition, the
  model's own uncertainty about what is where, explained before it is used: at one position, the class
  probabilities read out from the canvas as bars, peaked (confident) or flat (uncertain); entropy as the number for
  how flat; the entropy map over the scene; EG-C2F visiting next the tile of highest mean entropy; the policy reads the
  uncertainty, the canvas does not act.
- **Could:** the race over all of ADE20K validation; EG-C2F, whose viewpoints follow the uncertainty of the
  segmentation read out from the canvas, without reinforcement learning; RFS, more processing without new input, helping then declining.
- **Builds** [Yohaï, 2026-10-01: "we shouldn't go straight up to the animation, we should have a sort of grid with
  the colors and titles of each policy explaining what it is and why we [care], and then fade into the animation
  of them, and then perhaps show the race between them"]: first a grid, one cell per policy in its paper color, with
  its name, what it does and why it is in the paper, each with a small picture of its viewpoints; the cells then fade
  into the policies' animations on the scene, in the same places; then the race.
- **Shows:** the same scene under each of the paper's policies, glimpse boxes appearing one by one: R-IID and F-IID
  (random viewpoints, as in pretraining; F-IID starts from the full scene), C2F (a quadtree, coarse to fine, random
  order within a level), F2C (the same viewpoints, reversed), EG-C2F (C2F, visiting the most uncertain tile first),
  RFS (the full scene, repeated).
- **Says:** why each exists: R-IID and F-IID are the training distribution; C2F is the natural order; F2C isolates the
  order (same viewpoints by the end, worse result: what was seen changes how the next view is processed); EG-C2F
  shows that what the canvas holds can guide where to look, without reinforcement learning; RFS separates more processing from new
  input (it helps, then declines).
- **Data:** recorded bundles of one scene per policy (`canvit_pytorch.viz`, same seed).
- **Status:** draft (built: the six policies named over their first glimpses, then playing, then the race; the
  metacognition explanation to add).

### Uncertainty-based viewpoint selection
- **Title:** the paper's §6 ("EG-C2F's uncertainty-based viewpoint selection"), proposed 2026-10-01 after "The
  model's own uncertainty about what's where" was rejected [Yohaï: "terrible title for a slide. think of possible
  titles"]; alternatives offered: "Entropy-Guided Coarse-to-Fine" (the policy's name), "Guiding viewpoint selection
  with the canvas" (§6: "the ability of the canvas to guide viewpoint selection"). "The model's own uncertainty
  about what's where" and "metacognition" are said.
- **Must:** metacognition, explained: at a position, the class probabilities read out from the canvas, peaked or
  flat; entropy as how flat; the uncertainty map; EG-C2F averaging it per quadrant and looking next where it is
  highest, without reinforcement learning.
- **Could:** the next glimpses one by one; AME's attention-map entropy as the precedent (said).
- **Example:** the street of `#policies` after its first glimpse (continuity with the race); the surest and least
  sure canvas cells chosen by entropy (`throwaway/metacognition`).
- **Status:** draft (built).

## Results

### Benchmark results on ADE20K and ImageNet-1k
- **Title:** the paper's Fig. 3 caption.
- **Shows:** the history chart of `#history`, now with CanViT-B: 45.9% ADE20K mIoU (frozen, linear probe, C2F, 64²
  canvas) and 84.5% ImageNet-1k top-1 (fine-tuned).
- **Status:** ready.

### Active ADE20K segmentation
- **Title:** "Active ADE20K segmentation", subtitle "Accuracy v. efficiency" [Yohaï, 2026-10-01]. Before it, the
  paper's Figure 3A caption scoped to what the chart shows: active models only [Yohaï, on the caption alone: "ACTIVE.
  ACTIVE."], segmentation only; "A new active-vision state of the art in
  accuracy and efficiency" (the page's section) spoke of more than segmentation [Yohaï, 2026-10-01: "shorten that
  title ... this plot was ONLY about segmentation but you talk more generally... beware"]. No best-prior reference
  line [Yohaï: "remove the 'best prior active model' horizontal line"].
- **Builds** [Yohaï, 2026-10-01: "we should not show everything at once ... our worst policy at 32^2, then at 64^2
  canvas, then we show previous active models, then we show our best policy at both resolutions"]: F2C at 32²; F2C at
  64²; the prior active models (the cost axis widens to 835 GFLOPs as they arrive, pushing CanViT's curves left);
  EG-C2F at 32² and 64². `<canvit-frontier series=...>` set by fragments; each curve labeled at its end.
- **Shows:** `<canvit-frontier>`: 38.5% from one low-resolution glimpse of the full scene; even fine-to-coarse beats
  the prior models.
- **Must:** why cost matters (money and time per experiment), training cost first, the main concern for scientists
  [Yohaï, 2026-10-01]: what CanViT-B cost to train against the prior active models, each system's own training
  (`AGENTS.md`, rebuttal facts: hours, EFLOPs, glimpses, with their qualifiers); then inference: from a single glimpse
  of the full scene, CanViT-B beats the best prior active model with 20 times fewer inference FLOPs, and even F2C
  beats it; scoped to segmentation (the rebuttal: no efficiency claim on classification).
- **Could:** cost per glimpse and to train against an AdaGlimpse-style design on the same harness (22 against 212
  TPU chip-hours); the model running on a laptop or in a browser.
- **Says:** first why cost matters to this audience [Yohaï, 2026-10-01: "properly make the case of why anyone would
  [care] about efficiency (it's about money, making experiments cheap, etc)"]: compute is money and time; a
  model twenty times cheaper to run makes every experiment twenty times cheaper (more scenes, glimpses and policies,
  policies learned by trial and error) and runs on a laptop. Then: "Even our worst policy, a policy worse than random
  ... got better performance at lower cost."
- **Status:** draft: the case for cost is not on the slide yet (`PLAN.md`).

## Closing

### Toward action-aware neuro-AI research on vision
- **Title:** the UNIQUE application.
- **Shows:** three questions an active observer lets us ask: what is kept across views; how what was seen changes
  how the next view is processed; how human scanpaths compare with policies on the same observer.
- **Says:** limitations in passing (static scenes, a passive teacher); related work (Thorat et al. 2025, FOVI); "the
  first of many Active-Vision Foundation Models".
- **Status:** ready (visual to find).

### Quickstart
- **Title:** the README's heading.
- **Must:** that using CanViT takes a few lines anyone can read: install, load, glimpse, read the canvas
  [Yohaï, 2026-10-01: "the point is to show them it's trivial to start using this"].
- **Code** [Yohaï, 2026-10-01: "don't define both glimpses upfront, don't use a for loop, inline, and improve the viz /
  animation"; "NO LOOP PLEASE. ONE AFTER THE OTHER. DO NOT DEFINE BOTH UPFRONT."]: `quickstart.py`, the segmentation
  model, each glimpse written where it is taken; it runs as shown (`throwaway/quickstart/export.py` executes it). The
  second viewpoint, the left of the street, was chosen by `throwaway/quickstart/sweep.py`: it adds person and van.
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

### End-to-end policy learning (reserved)
- **Title:** to choose with the authors ("What about RL?" was floated as a spoken framing).
- **Must:** with a good observer, learning a policy becomes easy: a policy trained on top of the frozen CanViT-B
  (the VPE token exists for this) and how it compares with C2F and EG-C2F [Yohaï, 2026-10-01: "having the observer
  makes having a great policy easy (this WILL need work!)"].
- **Data:** CanViT-PyTorch-RL (unpublished) and any experiment run before the talk; nothing is shown before it is
  read and verified.
- **Status:** reserved.

### What's next (reserved)
- **Must:** teasers of the work that follows, as the authors decide [Yohaï, 2026-10-01: "we might want future /
  teasers etc."].
- **Status:** reserved.

### Canvas Attention
- **Title:** the paper's; "dumb, but not too dumb" is said.
- **Place:** Backup [Yohaï, 2026-10-01: "details like canvas attention might not be fully relevant, I don't know...
  maybe we don't even show them in the talk, or only as backup?"]: its point for this audience fits in a sentence,
  said on Canvas Vision Transformer architecture; the read–write pair and its projections are machine-learning detail.
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

### A general-purpose observer lets you use any policy
- **Title:** the project page's line; a bonus, in Backup [Yohaï, 2026-10-01: "no one [cares] enough for this
  to be the title of a main-talk slide, bonus at best. they care about science and what they can do with it"].
- **Shows:** `<canvit-live>`: I choose where to look; then EG-C2F chooses, from the uncertainty of the segmentation read out from the canvas.
- **Status:** ready (layout to fit; offline runtime in `PLAN.md`).
