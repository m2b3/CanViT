# MAIN 2026: positions and the deck

The talk's design as it stands. How to work is in `AGENTS.md` and `../AGENTS.md`; facts are in
`sources/checked-facts.md`. (Y) marks the authors' positions, (A) an agent's inference,
which is not a position until the authors confirm it. Nothing in the deck is approved unless the authors discussed it.

## The talk

An AI talk in a neuro-AI session: how to get good foundation models for action-aware research. The audience should
leave with a recipe: what belongs in the observer and what in the policy, policy-agnostic pretraining, and avoiding
reinforcement learning when one can (Y). Practicality runs through it: what works, not beautiful ideas that do not
work (Y). The paper (arXiv:2603.22570) is accepted at NeurIPS 2026.

Part I (Sabrina) argues why Active-Vision Foundation Models: deep networks as models of vision, DINOv3 as a vision
foundation model, what such models miss (gaze shifts, a partial high-acuity view), the history of active vision, the
gap between passive and active models, the observer and the policy, and what an Active-Vision Foundation Model needs.
Part II (Yohaï) presents CanViT through the paper's two contributions, each answering a problem: the architecture
(instantaneous vision and memory across viewpoints) and the pretraining (visuospatial intelligence for any task and
any viewing policy), then the experiments.

## Positions

Observer and policy:
- The distinction between observer (instantaneous vision and memory) and policy (action selection: where and at what
  zoom to look next) is one of the work's main conceptual moves and is made explicit, after the audience knows what
  active vision is and why they might care. Say "observer", never "vision encoder", which leaves out memory.
- "Before knowing where to look, we should know how to see" is the philosophy of policy-agnostic pretraining as a
  whole, not a lead-in to the learned policy. Prior active models did not yet see well, so the observer comes first.
- "The observer is the bottleneck, not the policy" goes in, with its nuances: on `#frontier`, even F2C's
  worse-than-random order beats the prior models, which the paper reads as perception, not viewpoint selection, being
  the bottleneck in that task.

Policy-agnostic pretraining:
- It is first a representation-learning problem, not a reinforcement-learning one. Training observer and policy
  together, done naively, works badly, for separate reasons:
  - Specificity: observer and policy co-adapt, even with a stop-gradient (what the observer sees is shaped by the
    policy, and the policy reads what the observer saw). The observer may then fail with other policies: a new policy
    (recorded human or monkey scanpaths, a better policy found later) comes with no guarantee.
  - Instability: policy gradients bring variance, instability, possibly collapse. Reinforcement learning is a last
    resort, not useless.
- Co-training (at the same time) differs from joint training (through shared gradients); a stop-gradient allows the
  first without the second. Joint training remains possible later: a policy-agnostic observer can initialize it.
- Task agnosticism motivates policy agnosticism: where people look depends on their goal (Yarbus), and a model does
  not know its future goals, so it should support any sequence of fixations. A general-purpose policy exists in one
  sense (free viewing), and a human's policy is conditioned on the goal; a model conditioned on any goal would need
  something like a future vision-language model, so a task-agnostic model is made policy-agnostic instead (for
  questions).
- Replaying recorded scanpaths through the observer is a payoff of policy agnosticism, said plainly and without
  over-claiming. Uniform-acuity glimpses are not a limitation of policy agnosticism.
- The deeper point is information density: knowing whether a look was good says little about the world, while
  predicting what one will see teaches about it. The parallel with language models (pretraining without
  reinforcement learning, post-training with it) is said on `#learned-policy`.
- On the learned policy, "cheap" is always qualified: cheap to train, and it works well.

Pretraining and its cost:
- The two problems: how to make the model robust to any policy (answer: policy-agnostic pretraining), and how to give
  it rich knowledge of the visual world without an enormous, costly and wasteful amount of computation (answer: dense
  latent distillation from DINOv3, which knows the scene seen whole; CanViT learns to build it from glimpses).
- Label-free at both levels: DINOv3 learned without labels, and CanViT learns from its features without labels.
- One should not waste resources on the wrong problem. The point is not that CanViT needs DINOv3: such models give
  task-agnostic representations, cost a great deal to train, are central to current research and to neuro-AI
  (Raugel et al.), and building on them is smart. `#pretraining`'s hours chart is right: repeating DINOv3's training
  would take a long time. Training efficiency matters scientifically; inference efficiency is a bonus.
- Precomputed teacher features can be divisive; the camera-ready says precomputation is optional, a tradeoff between
  storage and compute. Truncated backpropagation through time stays in the paper; generalizing to longer rollouts
  is a good result.

Part I's framing:
- Vision foundation models are introduced before Active-Vision Foundation Models, through DINOv3, which returns later
  (Raugel et al.; CanViT's teacher): image in, semantic features out, meaningful without adapting the model. PCA is
  signposted there as how the talk shows features.
- The key distinction among models of vision is supervised against self-supervised: learning from visual data alone,
  without imposing a human understanding through labels. "General", not "flexible". Self-supervision relaxes
  constraints.
- Raugel et al.'s result is shown and said for DINOv3, which they studied; representations that match higher cortex
  take training at scale. Their work is part of this work's inspiration: first build the model, then relate it to
  neural data; this work did the first.
- What passive models miss, in one breath: gaze shifts and a partial, high-acuity view; this ties to CanViT 2.
- Active vision's history in two acts: deep active computer vision since RAM, then neuro-AI active models.
  `#timeline` claims no completeness; goal-directed search models would be a category error beside it.
- Image-level (classification) against pixel-level (segmentation) tasks is made clear.

Results:
- Two angles kept apart: the differences between viewing policies, and peak performance against existing models.
  Each result appears once.
- CanViT at 21 glimpses beside prior models at 8 is fair: they stopped at 8 for diminishing returns.
- 0.4 mIoU over EG-C2F is a large difference against a heuristic that is already excellent.
- Accuracy is shown as top-1 accuracy, never error; ImageNet-1k's axis starts at 70% so the gaps show.

Reference frames and CanViT 2:
- CanViT's glimpses are uniform-resolution crops, not foveated; eccentricity-dependent acuity is part of CanViT 2,
  which appears in broad strokes only (acuity, dynamic scenes). Reference frames are a functional argument: the
  backbone works in glimpse (retinotopic) coordinates, the canvas in scene (spatiotopic) coordinates; the
  neuroscientific plausibility of a spatiotopic representation is set aside, with no caveat about eye-centered frames.

Presentation:
- The title slide names the speakers; `#paper` names the whole team and NeurIPS 2026, where the audience first hears
  "CanViT".
- The paper's Figure 1 opens CanViT, before the architecture, built from its own code, with no title.
- Never say CanViT saw "an order of magnitude more" data than previous active models.
- No credit for third-party video on slides.
- Be ready for another meaning of "active-vision foundation model": a model given a task in language that selects
  where to look, as I-Perceive does.

## Agent inferences

- (A) How the contributions fit (paper §4–5): glimpses from arbitrary viewpoints cannot be carried forward as a grid,
  so CanViT keeps memory in a canvas fixed in scene coordinates; that makes dense outputs a token-wise readout and lets
  gradients flow across time. The canvas is large, so it has no self-attention or MLP and Canvas Attention projects
  only on the glimpse side. Pretraining asks, after every glimpse, for the teacher's features of the whole scene:
  dense in space (unseen regions too) and in time (a loss every step, so short truncated chunks with constant
  memory). A canvas in scene coordinates matches the teacher's patch grid token for token.
- (A) Prior work's own ablations suggest that choosing where to look added about 0 to 3
  points on natural images with a low-resolution glance, many glimpses or dense segmentation, and 5 to 70 points where
  information is small, sparse and off-centre; within a paper, observer changes often outweighed policy changes. Too
  zoomed in for the talk (Y).
- (A) The dense learning signal is not why CanViT trains without reinforcement learning: it could be trained without
  it with a sparse signal. What density buys has not been worked out with the authors.

## The deck as it stands

Speaker notes are in `index.html`; who wrote them is under "Notes provenance". Part II's notes follow the speakers'
checklist (`../AGENTS.md`); its measured rate is about 128 words per minute.

Part I (Sabrina):
- `#title`: the speakers and affiliations on the brand gradient, beside a silent loop of CanViT on four scenes, five
  glimpses each (`experiments/title_teaser`: ADE_val_00000419, 00001882, 00000203, 00001860, chosen by eye among the
  scenes whose glimpses change the canvas most).
- `#tradition` "Deep neural networks as models of vision": Yamins et al. 2014 alone; Brain-Score (Schrimpf et al.
  2018); its zoom on models above 70% ImageNet top-1, where the link no longer holds; Yamins beside Zhuang et al. 2021,
  under "Supervised" and "Unsupervised / self-supervised".
- `#foundation` "DINOv3": a living room and its features (PCA), the linear decoding, then detection and depth from
  task heads on frozen features.
- `#dinov3-and-the-brain`: Raugel et al.'s study, then their figure on alignment with higher cortex.
- `#passive-models`: "What's missing?".
- `#human-vision`: foveated glimpses with a jumping fixation cross, then the street at full resolution.
- `#bandwidth`: the world against sensory bandwidth over computational resources, tipped on a click.
- `#timeline` "Deep active computer vision": RAM, then the later models, then the neuro-AI models.
- `#gap` "The wide gap between passive and active computer vision": ADE20K segmentation introduced on an image and its
  mask, then the active models' bars, then the passive models' (`gap.js`).
- `#active-vision-model` "What's in an active-vision model?": instantaneous vision, memory and action selection, as
  observer and policy; prior active work ringed around action selection, passive foundation models around
  instantaneous vision.
- `#task-dependent-scanpaths` "Where people look depends on their goal": Yarbus's painting, then the records.
- `#compare` "Toward Active-Vision Foundation Models": passive foundation models and prior active models, then the
  goal column.

Part II (Yohaï):
- `#paper`: the team and NeurIPS 2026.
- `#canvit-rollout`: the paper's Figure 1, a glimpse at a time (`experiments/canvit_rollout`).
- `#outline-architecture`: Architecture, Pretraining, Experiments; the two problems said over them.
- `#vit-on-a-glimpse`: a zoomed glimpse of a lamp as white as its ceiling, whose patch tokens set it apart
  (`experiments/vit_glimpse`: ADE_val_00001517, chosen by eye from a sweep of objects their pixels barely separate;
  alternatives in `examples.py`).
- `#integrating-information-across-viewpoints`: separate glimpses leave the scene unintegrated and unplaced.
- `#architecture-title`, `#architecture`: the paper's figure, full height; backbone and canvas, reads and writes.
- `#sr-rope`: a glimpse patch and a canvas patch in scene coordinates; zoom at a fixed center, then a move.
- `#canvas-attention-title`, `#canvas-attention`: the ViT blocks ringed (self-attention and MLPs on the glimpse side
  only), then one step of Canvas Attention and its cost (`experiments/canvas_attention`; panel C hidden by `hide-c`).
- `#outline-pretraining`, `#how-to-see-first`: the principle over Part I's diagram, the observer ringed.
- `#pretraining-policies`: R-IID and F-IID.
- `#distillation` "Policy-agnostic dense latent distillation": the whole scene in DINOv3's features after every
  glimpse.
- `#pretraining` "Pretraining CanViT-B": 166 H100-hours, then DINOv3 7B's, on a rescaling axis.
- `#table-callback`: the table's two corners, filled.
- `#outline-experiments`, `#choosing-where-to-look`: Part I's diagram, the policy ringed.
- `#c2f-and-f2c`: each policy's quadtree levels in order.
- `#eg-c2f`: the quadrants ranked by entropy, then visited in that order.
- `#policy-accuracy` "Accuracy by viewing policy": ADE20K and ImageNet-1k, frozen then fine-tuned, with 95% bands for
  stochastic policies.
- `#frontier` "ADE20K accuracy against inference compute": F2C at 32², then 64², then EG-C2F, with the single
  glimpse's 20× fewer FLOPs and +10.9 mIoU against AME (SETR) drawn in the chart.
- `#state-of-the-art` "A new active-vision state of the art": `#gap`'s chart with CanViT-B's 45.9, then ImageNet-1k
  (`gap.js`).
- `#learned-policy` "Learning where to look": Q-learning against the heuristics over five glimpses, then 640K glimpses
  of policy learning against 1B of pretraining.
- `#quickstart` "Open-source and easy to use": the code walked through (`code-only` hides its scene and maps).
- `#next` "Toward CanViT 2": a glimpse moving over a still street, beside CanViT 2's foveated view of a video.
- `#links`: QR codes and funders; up during questions.

Appendix, for questions: `#canvas-updates` (the emergent hierarchy within a glimpse, `experiments/canvas_updates`),
`#rollout` (the live model), `#extrapolation`, `#detail`, and the paper's figures and tables listed in
`sources/paper-figures.json`, each with the question it answers in its notes.

## Notes provenance

The authors' wording except where noted. Agent drafts stay drafts until an author rewords them.
- Part I (`#title` to `#compare`): Sabrina's. Agent lines: two on `#dinov3-and-the-brain` (the scaling suggestion,
  said as written); `#tradition`'s 70% line; `#bandwidth`'s draft, except its last line, which is a co-author's.
  Yohaï's: `#tradition`'s "The same trend was found in later work, at least up to a point".
- Part II (`#paper` to `#links`): Yohaï's, from his run-throughs, false starts removed. Agent lines: the recurrent
  CLS token on `#architecture`; on `#sr-rope`, the canvas patches' fixed coordinates and the zoom lines; on `#eg-c2f`,
  the ranking and visiting lines; `#choosing-where-to-look`'s line; `#quickstart`, `#next` and `#links`.
- Appendix: `#canvas-updates` is Yohaï's up to its first Write; the rest of the appendix's notes are agent drafts.

## Rehearsal timings

- Part I, 2026-10-25, about 6:15. Part II, 2026-10-27, 16:37 session, per slide (s): `#paper` 29, `#canvit-rollout` 76,
  `#outline-architecture` 18, `#vit-on-a-glimpse` 36, `#integrating-information-across-viewpoints` 50,
  `#architecture` 98, `#sr-rope` 70, `#canvas-attention` 95, `#outline-pretraining` with `#how-to-see-first` 29,
  `#pretraining-policies` 43, `#distillation` 78, `#pretraining` 58, `#table-callback` 19, `#state-of-the-art` 41,
  `#c2f-and-f2c` 72, `#eg-c2f` 90, `#policy-accuracy` 142, `#frontier` 74, `#learned-policy` 65.
- Full run, 2026-10-27, 19:56: 21:29 (Part I 6:17, Part II 15:12) for a 25-minute slot
  (`sources/rehearsal-2026-10-27-1956.json`, from `experiments/rehearsal_timing.py`).
- Each slide's notes open with its budget (the 25 minutes shared in proportion to the notes' words, computed by
  `deck.js`) and its range in the measured run.

## For questions

45.9 against the teacher's 47.2 takes 21 glimpses at a 64² canvas (about 460 GFLOPs against 215); on
ImageNet-1k, CanViT-B leads on peak accuracy, not efficiency (camera-ready `tab:in1k-efficiency`); `#rollout`'s meter
shows one scene's mIoU (18 to 30% on the street), which counts every class predicted there, unlike the benchmark.
