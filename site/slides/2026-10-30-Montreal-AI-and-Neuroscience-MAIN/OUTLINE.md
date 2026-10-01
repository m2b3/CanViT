# MAIN 2026 talk: outline

The talk, designed slide by slide before it is built (`../AGENTS.md`, "Workflow"). The framings, scoped claims and
decisions it relies on are in `AGENTS.md`; the sources in `sources/`. Status per slide: **ready** (content and
visual exist), **build** (the visual must be built from data we have), **data** (needs an export or an experiment),
**figure** (a third-party figure, located), **decide** (the authors' call).

Revision: draft 1, 2026-10-01, Claude Code. Not yet reviewed.

## The argument

Each slide states or proves one step. The opening order (tradition, then vision is active) is Yohaï's.

1. Deep networks became models of the visual system.
2. Human vision is fundamentally active.
3. The vision models used to model the brain are passive; this limits their value as models of natural, active
   vision.
4. Seeing goes beyond what is sampled: prior knowledge of the world lets an active observer infer the unseen.
   Machines get such knowledge at scale, from foundation models.
5. Active computer vision never reached that scale, and has struggled to match passive vision.
6. The observer was the bottleneck: no policy can make up for a poor observer.
7. Goal: a task- and policy-agnostic Active-Vision Foundation Model.
8. An architecture built to scale: CanViT.
9. A learning signal built to scale: passive-to-active dense latent distillation.
10. Scale reached, on a single GPU.
11. It works.
12. An observer for any (tested) policy.
13. What it opens for neuro-AI.

**Where I (Claude Code) stand on it.** I agree with the chain, with three scopings the slides must carry:
step 3 holds for natural vision with eye movements (core-object-recognition paradigms exclude them on purpose,
and the models fit those paradigms well); step 4 needs both the evidence that scale brings general visual
knowledge and the evidence that more ImageNet accuracy stops making models more brain-like (Linsley et al. 2023),
so the claim is about knowledge needed to understand scenes, not about brain similarity; step 5's reasons (RL,
architectures that do not scale) are our interpretation, so they are stated as such. Step 6's "no policy can make
up for a poor observer" is the paper's sentence with its condition (perfect observer, static scene, enough
glimpses); the evidence is that CanViT beats prior models even with a worse-than-random order.

## Part I: modeling vision (steps 1–3)

### 1. Title
- **On screen:** the CanViT wordmark; "Toward Active-Vision Foundation Models"; "Canvas Vision Transformer";
  Yohaï-Eliel Berreby · Sabrina Du · Audrey Durand · B. Suresh Krishna; McGill, Mila and Université Laval logos;
  NeurIPS 2026 logo; "MAIN 2026 · Montréal · October 30, 2026".
- **Visual:** the wordmark is the visual. Option: a CanViT rollout looping quietly beside it (canvas filling in),
  without explanation, to be recognized later.
- **Say:** "This is CanViT, the Canvas Vision Transformer: the first task- and policy-agnostic Active-Vision
  Foundation Model. Joint work with Sabrina Du, Audrey Durand and Suresh Krishna."
- **Status:** build (logos in hand: `scratchpad/logos`, to install with provenance).

### 2. Deep networks as models of the visual system
- **Step:** 1.
- **On screen:** a lineage, left to right, one figure per click: receptive fields in cat visual cortex (Hubel &
  Wiesel 1962) → the hierarchy of visual areas (Felleman & Van Essen 1991) → a hierarchical network inspired by
  them (Neocognitron, Fukushima 1980) → task-optimized networks predict neural responses in IT (Yamins et al.
  2014) → Brain-Score (Schrimpf et al. 2018). Each figure small, cited under it.
- **Visual:** classic figures from the papers (figure agent; files in `assets/figures/`).
- **Say:** "Modeling vision with neural networks has a long tradition. Hubel and Wiesel found receptive fields
  organized in a hierarchy; Felleman and Van Essen mapped the hierarchy of visual areas; Fukushima built a
  hierarchical network inspired by it. Decades later, deep networks trained to recognize objects turned out to
  predict neural responses along the ventral stream better than models designed by hand, and Brain-Score now ranks
  hundreds of networks by how brain-like they are. Today, vision foundation models are among the models used to
  model brain function."
- **Keywords:** receptive field, hierarchy, ventral stream, task-optimized (goal-driven) models, neural
  predictivity.
- **Sources:** `sources/citations.md`; Hubel & Wiesel 1962 (J. Physiol.); Felleman & Van Essen 1991 (Cereb.
  Cortex); Fukushima 1980 (Biol. Cybern.); Yamins et al. 2014 (PNAS); Schrimpf et al. 2018 (bioRxiv); for the last
  sentence, a study using a vision foundation model as a brain model (literature agent: e.g. Raugel et al. 2025).
- **Status:** figure (pending the figure agent).

### 3. Human vision is fundamentally active
- **Step:** 2.
- **On screen:** a natural scene seen through a fovea: sharp at the fixation, coarser with eccentricity; the
  fixation jumps about three times per second; past fixations leave a fading trace. One line under it: "We shift our
  gaze about three times per second, orienting the high-resolution fovea toward regions of interest." Cite under.
  Optional click: the acuity-against-eccentricity curve as an inset.
- **Visual:** `<canvit-foveate>` (to build; shared with the project page): a blur pyramid whose level grows with
  distance from the fixation, following a published acuity falloff (Strasburger et al. 2011's linear MAR scaling);
  the scanpath from a recorded bundle's viewpoints, or hand-placed fixations on salient objects. The presenter's
  mouse can take over the fixation.
- **Say:** "Human vision is fundamentally active: we shift our gaze multiple times per second, orienting our
  high-resolution fovea toward regions of interest, and integrate information from multiple viewpoints into a
  coherent understanding of our surroundings' spatial and semantic structure."
- **Keywords:** gaze shift, fovea, acuity, regions of interest, integration.
- **Sources:** Henderson & Hollingworth 1999 (fixations shift about three times per second); Strasburger et al.
  2011 (acuity falloff); Curcio et al. 1990 (cone density).
- **Status:** build.

### 4. Active vision is sequential
- **Step:** 2.
- **On screen:** left, Yarbus's recordings of one observer looking at Repin's painting under seven instructions;
  right, three lines appearing one per click: strategic planning · integration of evidence across time in visual
  working memory · top-down recurrent feedback, each with its citations.
- **Visual:** `assets/yarbus-the-visitor.jpg` (Wikimedia Commons, public domain; data from Yarbus 1967), or the
  book's own figure (figure agent).
- **Say:** "This process is inherently sequential. Where we look depends on what we are trying to do: here, one
  observer looks at the same painting under seven different instructions. We integrate evidence across fixations
  in visual working memory. And signals from higher areas condition early visual processing, so that each fixation
  is processed in light of what was seen before, rather than from a blank slate."
- **Keywords:** sequential, strategic planning, visual working memory, top-down recurrent feedback.
- **Sources:** Yarbus 1967; Hoppe & Rothkopf 2019; Baddeley & Hitch 1974; Melcher 2001; Rao & Ballard 1999; Gilbert &
  Li 2013; Kar et al. 2019 (the paper's own citations for these three points).
- **Status:** ready (figure in hand).

### 5. Vision foundation models lack active sensing
- **Step:** 3.
- **On screen:** two columns. Human vision: glimpses over time, memory, feedback (the three points of "Active vision is sequential").
  A vision foundation model: one whole frame in, a feature map out, the next frame processed from scratch: no
  glimpses, no memory, no top-down recurrent feedback.
- **Visual:** a small animation: frames entering a passive encoder one after another, nothing carried over; beside
  it, glimpses accumulating into one representation. Drawn in SVG; talk-only.
- **Say:** "Leading vision foundation models used to model brain function lack the active sensing and top-down
  recurrent feedback of biological vision. They process each frame independently and passively, all at once. This
  limits their value as models of natural human vision, which fundamentally relies on eye movements. Rothkopf and
  colleagues put it bluntly: models of vision need some action."
- **Scope:** core object recognition paradigms present images briefly, at fixation; the claim concerns natural
  vision with eye movements.
- **Keywords:** passive, single-shot, uniform resolution, no memory, action-first models.
- **Sources:** paper §1; `sources/unique-travel-award.md`; Rothkopf, Bremmer, Fiehler & Dobs 2023 (Behav. Brain
  Sci., "Models of vision need some action").
- **Status:** build.

## Part II: what seeing needs (step 4)

Step 4, restated [Yohaï, 2026-10-01: "WHAT ABOUT NEURO? WHAT ABOUT COGNITION? WHY DO WE [CARE]?"]: seeing goes
beyond what is sampled. Looking at two opposite corners of a table, you do not need to look at its center to know
there is more table there: you know tables. Prior knowledge of the world is what lets an active observer sample
sparsely and still understand the scene. Machines get that knowledge at scale, from foundation models; active
machines never had it.

### 6. Perception goes beyond the information sampled
- **Step:** 4.
- **On screen, built click by click** [Yohaï, 2026-10-01: the opening question is demonstrated visually, with an actual
  table picture and localized crops, showing that moving retinotopic predictions into a scene-wide, spatiotopic map is
  not enough, because there is no extrapolation]:
  1. A real scene with a table; two small glimpses at its two ends light up; the rest of the scene darkens.
  2. Each glimpse, in its own (retinotopic) frame, beside the scene, with a passive model's segmentation of it: DINOv3
     ViT-B with its linear probe, on the glimpse alone.
  3. The two segmentations fly to where their glimpses were taken, onto a scene-sized (spatiotopic) map: the map is
     blank everywhere else. The middle of the table stays empty: pasting views into place does not extrapolate.
  4. Pause, then the question: what is between the two glimpses? You know: more table, because you know tables.
  5. Three short lines, one per click: perception is inference from prior knowledge · scenes have a grammar, what goes
     where · knowledge tells the eyes where they need not look.
- **Visual:** a component (to build) driven by a recorded example: the scene, the two viewpoints, DINOv3's per-glimpse
  segmentations (export: new, `canvit_pytorch.viz`), animated from glimpse frame to scene frame. The example comes from
  the table-corners sweep. The same scene and glimpses return in "Extrapolation to unobserved regions", where CanViT
  fills the middle.
- **Say:** "Two glimpses of a scene, at the two ends of a table. A model that looks at each glimpse on its own can say
  what is in it: here, table. Put each answer back where the glimpse was taken, and you get a map of the scene: with a
  hole in the middle. Moving local answers into scene coordinates is not enough. Yet you know what is between them:
  more table, because you know tables. Perception goes beyond the information given: it is inference from prior
  knowledge of the world. We know what scenes contain and where things go, and that knowledge guides where we look,
  and where we do not need to. This is what makes sparse, active sampling work."
- **Keywords:** retinotopic, spatiotopic, extrapolation, prior knowledge, inference, scene grammar, contextual guidance.
- **Sources (to verify):** Helmholtz 1867 (unconscious inference); Bruner 1957 ("Going beyond the information given");
  Kersten, Mamassian & Yuille 2004 (Annu. Rev. Psychol., object perception as Bayesian inference); Biederman, Mezzanotte
  & Rabinowitz 1982 (scene violations); Bar 2004 (Nat. Rev. Neurosci., visual objects in context); Oliva & Torralba
  2007 (TICS, context in object recognition); Torralba, Oliva, Castelhano & Henderson 2006 (Psychol. Rev., contextual
  guidance of eye movements); Henderson & Hayes 2017 (Nat. Hum. Behav., meaning maps); Võ, Boettcher & Draschkow 2019
  (Curr. Opin. Psychol., scene grammar).
- **Status:** data (sweep running) + build.

### 7. Foundation models
- **Step:** 4: where machines get world knowledge.
- **On screen:** "Trained once, at scale, without labels; adapted to many tasks." One frozen backbone, many readouts
  from the same features (segmentation, depth, classification, correspondence).
- **Say:** to write from `sources/foundation-models.md`: what a foundation model is (Bommasani et al. 2021); that
  vision foundation models acquire knowledge of the visual world (3D structure, correspondence, objects) without
  labels; that they are now among the models used to model the brain.
- **Status:** pending the literature agent.

### 8. DINOv3 feature maps
- **Step:** 4 (what a vision foundation model computes; the teacher of step 9).
- **On screen:** the explorer, built in stages: a photograph with its 16-px patch grid → one patch highlighted, its
  representation a vector of 768 numbers → hover: every patch colored by its cosine similarity to the hovered one
  → the three main directions of variation as colors (PCA) → a single linear layer turns the same vectors into a
  segmentation; for an object, the probability of its class as a heat map.
- **Visual:** `<canvit-features>` (to build) on DINOv3 ViT-B/16 features of the scene of "Perception goes beyond the information sampled" and one or two others
  at 512 px (32 × 32 patches), exported by `canvit_pytorch.viz` (new: teacher features and the DINOv3 ADE20K probe).
- **Say:** "Here is what such a model computes. DINOv3 cuts the image into patches and turns each into a vector,
  computed in the context of the whole image. Point at a patch, and the patches most similar to it light up: the parts
  of the same object, objects of the same kind. A single linear layer on the frozen vectors reads out a segmentation.
  None of this was learned from labels."
- **Scope:** PCA colors are not class labels; the linear probe is trained with labels, the features are not.
- **Keywords:** patch, feature map, embedding, cosine similarity, PCA, linear probe, self-supervised.
- **Status:** data + build.

### 9. Visual knowledge emerges at scale
- **Step:** 4.
- **On screen:** the evidence that general visual knowledge improves with the scale of data and models, and its
  limit for brain modeling: past a point, more ImageNet accuracy does not make models better models of
  inferotemporal cortex.
- **Say:** to write from `sources/foundation-models.md`, ending on: "Rich semantic visual priors only emerge at
  scale. An observer that is to infer what it has not seen needs them."
- **Status:** pending the literature agent.

## Part III: active vision (steps 5–7)

### 10. Active computer vision
- **Step:** 5.
- **On screen:** a chart, x = year (2012–2026), y = ImageNet-1k top-1. Click by click, the active models appear:
  the Recurrent Attention Model (2014, digits: marked below the axis), Saccader (2019), GFNet (2020), AdaGlimpse
  (2024), AdaptiveNN (2025).
- **Visual:** a year-by-performance chart component (to build) reading `sources/sota-history.json` (data agent).
  Possibly a second panel: ADE20K mIoU, where the active models are AME (2023) and AdaGlimpse (2024).
- **Say:** "Machines that look through glimpses have been built since the 1980s [active perception, animate
  vision]. In deep learning, the Recurrent Attention Model of 2014 learned with reinforcement learning where to look
  on digits. It took until 2019 for an active model, Saccader, to reach 75 percent on ImageNet; AdaptiveNN reached
  82 in 2025."
- **Sources:** `sources/citations.md`; `sources/sota-history.json`.
- **Status:** data (agent running) + build.

### 11. Active models have struggled to match passive ones
- **Step:** 5.
- **On screen:** the same chart; the passive models appear: the best published ImageNet top-1 each year, from AlexNet
  to DINOv3. The gap is visible. No CanViT.
- **Say:** "Over the same years, passive models went from AlexNet to foundation models. Despite their theoretical
  advantages, active models have struggled to match the accuracy and representational richness of their passive
  counterparts, and the disconnect is particularly striking on dense prediction. They stayed small and
  task-specific: trained with reinforcement learning, which is sample-inefficient, on architectures not designed to
  scale, they never reached the scale at which passive models acquired their knowledge."
- **Scope:** the reasons are our interpretation; the gap is measured.
- **Status:** data + build.

### 12. Three axes of an active vision model
- **Step:** 6.
- **On screen:** the project page's diagram: Seeing (instantaneous vision, memory) = observer; choosing where to
  look (action selection) = policy. Click: "Prior work focused here" on action selection. Click: "No policy can make
  up for a poor observer."
- **Visual:** one component shared with the project page (its "What's in an active-vision model?" section), with
  the cycling policy slot.
- **Say:** "An active vision model can be considered along three axes: making sense of what it sees at a given
  moment, updating a persistent understanding of the scene, and deciding where and at what zoom level to look next.
  The first two make an observer; the last is a policy. Prior work has often focused on the policy, often with
  reinforcement learning. But given a perfect observer, naive and strategic policies reach the same accuracy after
  enough glimpses of a static scene, and no policy can make up for a poor observer."
- **Status:** build (component from the page's section).

### 13. A task- and policy-agnostic Active-Vision Foundation Model
- **Step:** 7.
- **On screen:** the observer box filled with "CanViT"; the policy slot cycling through policies and landing on
  "any". Line: "Learning to see through glimpses, disentangled from learning where to look."
- **Say:** "So we set out to build the observer first: a vision model that understands the spatial and semantic
  structure of scenes across arbitrary sequences of glimpses, with representations that transfer across tasks and
  viewing policies. Disentangling learning to see through glimpses from learning where to look frees pretraining
  from reinforcement learning, and lets it scale."
- **Status:** build (same component, second state).

## Part IV: CanViT (steps 8–10)

### 14. Scenes, viewpoints and glimpses
- **Step:** 8 (the setting).
- **On screen:** a scene; a glimpse box moves and zooms; beside it, the 128 × 128 px the model receives.
  "Outside the glimpse, nothing."
- **Visual:** `<canvit-scene>` and `<canvit-map layer="crop">` stepped by fragments (exists), or the live model's
  scene panel with a scripted zoom.
- **Say:** "A glimpse is a fixed-resolution crop at a viewpoint: a position and a zoom level. Zoomed out, the model
  sees the whole scene, coarsely; zoomed in, a small region in detail. And unlike our peripheral vision, outside the
  glimpse there is nothing at all, not even a blur: a world of difference."
- **Status:** ready (components exist).

### 15. The Canvas Vision Transformer
- **Step:** 8.
- **On screen:** the architecture built click by click: the glimpse enters a ViT backbone (few tokens, deep:
  retinotopic) → a canvas tiles the scene (many tokens: spatiotopic) → reads: the backbone queries the canvas (top-down
  feedback) → writes: the canvas queries the backbone → after every glimpse, a linear readout of the canvas answers
  "what is where?" for the whole scene. Last click: the paper's figure.
- **Visual:** an animated SVG diagram (to build; candidate for the project page), tokens drawn as small squares in
  glimpse blue and canvas red, pulses along read and write.
- **Say:** "CanViT pairs a retinotopic Vision Transformer backbone, which processes each glimpse, with a spatiotopic
  latent workspace, the canvas. The canvas is a scene-wide memory, which can function as a cognitive map of the
  scene: a stable spatial reference frame. The backbone reads from it, so each glimpse is processed in the context of
  everything seen so far: top-down recurrent feedback. And it writes to it. At any moment, for any position, we can
  ask the canvas: what does the model think is there?"
- **Scope:** "cognitive map" in the sense of a spatially organized scene representation.
- **Status:** build.

### 16. Scene-Relative RoPE
- **Step:** 8.
- **On screen:** the canvas's fixed grid of patch centers; the glimpse's 8 × 8 patch centers in the same scene
  coordinates, clustered when zoomed in, spread when zoomed out, moving with the viewpoint.
- **Visual:** small component (to build) driven by a recorded bundle's viewpoints.
- **Say:** "Both streams use the scene's coordinates: every glimpse patch knows where it is in the scene, and how
  zoomed in it is, from how tightly its patches are packed. That shared reference frame binds the retinotopic and the
  spatiotopic streams."
- **Scope:** the geometry is given to the model (each glimpse's position and scale); CanViT does not model how the
  brain computes remapping.
- **Status:** build.

### 17. Canvas Attention
- **Step:** 8 (why the memory can be large: the link to scale).
- **On screen:** read and write as two arrows; the canvas side carries no learned weights: "dumb, but not too dumb".
  One number, small: a read–write pair on a 64 × 64 canvas costs 2.8 GFLOPs instead of 37.3 with projections on
  both sides.
- **Say:** "The memory is deliberately simple: learned projections only on the glimpse side; canvas tokens never go
  through an MLP or self-attention. That is what makes a large, fine-grained canvas affordable, and what lets us query
  it after every glimpse. Earlier dense active models paired a cheap encoder with a decoder costing far more at every
  step."
- **Status:** build (simple diagram).

### 18. Passive-to-active dense latent distillation
- **Step:** 9.
- **On screen:** the scene of "DINOv3 feature maps" again. Left, DINOv3's feature map of the whole 512-px scene (its PCA colors).
  Right, CanViT's prediction of it from the canvas, after each glimpse, in the same basis and color limits, with an
  error map and the cosine similarity under it.
- **Visual:** new export (`canvit_pytorch.viz`): the pretrained readout's prediction per glimpse, destandardized,
  projected on the teacher's PCA basis; displayed with `<canvit-rollout>` and new layers.
- **Say:** "How do we teach it? We take DINOv3, give it the whole scene at high resolution, and keep its feature
  map. CanViT only gets glimpses, and after every glimpse it must predict that feature map for the whole scene,
  including what it has not seen. This rich learning signal lets CanViT inherit much of its teacher's world
  knowledge, while learning to refine its understanding glimpse by glimpse."
- **Status:** data + build.

### 19. The teacher defines what to reconstruct
- **Step:** 9.
- **On screen:** partial observations → prediction of g(whole scene). Pixels: g = identity (masked autoencoders;
  AdaGlimpse). CanViT: g = DINOv3. The same scene shown both ways.
- **Say:** "Is it cheating to learn from a model that sees everything? Reconstructing pixels from partial views is
  distillation too: from the identity. Every reconstruction objective has a teacher; we chose one whose space
  encodes meaning rather than texture. The teacher tells the observer what to preserve and what to infer. The student
  only ever sees glimpses, and no labels are used anywhere."
- **Status:** build.

### 20. Policy agnosticism
- **Step:** 9.
- **On screen:** rollouts sampled live as in pretraining: random positions and zoom levels (small glimpses
  favored), random lengths; one rollout per scene starts from the full view.
- **Say:** "No policy is baked in: every training rollout uses random viewpoints, random zoom levels and a random
  length."
- **Status:** build (small sampler, the paper's distribution).

### 21. Pretraining CanViT-B on a single H100
- **Step:** 10.
- **On screen:** 13.2 M scenes, an order of magnitude more than previous active models · 1 B glimpses · 166 hours on
  one H100. Under it: the teacher's features, computed once, about 8 H100-equivalent hours.
- **Say:** "Because the architecture is cheap and the signal needs no policy, active-vision pretraining reaches
  foundation-model scale on a single GPU."
- **Status:** ready (numbers from the paper).

## Part V: evidence (steps 11–12)

### 22. A CanViT rollout
- **Step:** 11.
- **On screen:** `<canvit-episode>` on recorded ADE20K scenes: the glimpse, CanViT, the canvas decoded into a
  segmentation after every glimpse, including regions not yet seen.
- **Status:** ready.

### 23. The canvas carries what was seen
- **Step:** 11.
- **On screen:** a smooth path of viewpoints over a scene; two segmentations side by side: canvas kept, canvas
  reset at every glimpse.
- **Visual:** `<canvit-path>` on the recorded path bundles (exists).
- **Say:** "Same glimpses, two conditions: keep the state, or reset it at every glimpse. With memory, what was seen
  stays and improves; without it, the model only knows its current view."
- **Scope:** the reset clears the whole recurrent state (canvas and CLS token).
- **Status:** ready.

### 24. Extrapolation to unobserved regions
- **Step:** 11 (the payoff of "Perception goes beyond the information sampled").
- **On screen:** the same scene and the same two glimpses at the two ends of the table. Three panels: the scene with
  the glimpse boxes; DINOv3 run on each glimpse alone, its segmentation pasted where the glimpse was taken, nothing
  elsewhere; CanViT's segmentation decoded from the canvas after the two glimpses. Click: the probability of "table"
  from the canvas, as a heat map: the table's shape continues through the unseen middle.
- **Visual:** recorded from the table-corners sweep (`throwaway/table_corners`: conditions A, B, A then B; measured over
  every large object in ADE20K validation, so the example is not cherry-picked without saying so).
- **Say:** "A passive model, given these two glimpses, can only describe what is inside them. CanViT, having seen
  both ends, predicts the table between them, though it never looked there."
- **Scope:** the middle was never inside a glimpse (no full-scene glimpse first); report the sweep's average, not only
  the best example.
- **Status:** data (sweep running).

### 25. Results on ADE20K and ImageNet-1k
- **Step:** 11.
- **On screen:** the year chart of "Active computer vision", now with CanViT-B: 45.9% ADE20K mIoU (frozen, linear probe) and
  84.5% ImageNet-1k top-1 (fine-tuned). Under it: 38.5% from a single low-resolution glimpse of the full scene, versus
  27.6% for the best prior active model.
- **Status:** build (after "Active computer vision").

### 26. Architecture and pretraining both matter
- **Step:** 11.
- **On screen:** the control: same DINOv3 distillation on ImageNet-1k, same glimpse budget: ADE20K mIoU after 4
  glimpses, CanViT-B 33.7 vs AdaGlimpse 15.6, at about a tenth of the training compute.
- **Scope:** does not quantify the teacher's own contribution to absolute performance.
- **Sources:** rebuttal Table R1 (reviewer DjZB), promised for the camera-ready.
- **Status:** build.

### 27. Viewing policies at inference
- **Step:** 12.
- **On screen:** mIoU by glimpse for C2F, F2C and RFS (`ade20k_seg.json`): same viewpoints in opposite orders end at
  different accuracies; repeating the full view helps, then hurts.
- **Say:** "Processing order matters: coarse-to-fine beats fine-to-coarse although both see the same viewpoints by
  the end. Looking again at the same view first helps, then declines without new viewpoints."
- **Scope:** order sensitivity under this protocol; no mechanism claimed.
- **Status:** build.

### 28. CanViT-B, live
- **Step:** 12.
- **On screen:** the model running in the browser. Purposeful actions: look at an object, move away, come back,
  reset; then let the canvas's uncertainty choose (EG-C2F).
- **Say:** "I chose those views; CanViT integrated them. Now the canvas chooses."
- **Status:** ready (`<canvit-live>`); a slide-sized layout and offline runtime to do.

### 29. Learned viewing policies
- **Step:** 12.
- **Status:** decide (unpublished).

## Part VI: closing (step 13)

### 30. Toward action-aware neuro-AI research on vision
- **On screen:** questions this model lets us ask: what should be retained across views; how context changes
  interpretation; which observations are worth acquiring; how human scanpaths compare with policies on the same
  observer. Limitations underneath: static scenes; geometry given; a passive teacher.
- **Say:** "We hope CanViT will serve the neuroscience and AI communities alike as the first of many Active-Vision
  Foundation Models."
- **Status:** build.

### 31. Code and models
- **On screen:** `pip install canvit-pytorch`; GitHub, Hugging Face, project page with a QR code; PyTorch, and JAX and
  MLX when ready; runs in the browser.
- **Status:** build (logos in hand; QR to generate).

### 32. Acknowledgments
- **On screen:** co-authors; collaborators (to confirm); funders from the paper's acknowledgments, as logos.
- **Status:** decide (lists to confirm).
