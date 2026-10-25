# The MAIN 2026 talk: positions, open questions, the deck, open work

The talk's strategy, which changes as the authors and agents iterate. How to work is in `AGENTS.md`; facts are in
`sources/checked-facts.md` and `sources/literature.md`.

Markers:
- **(Y)**: a position Yohaï stated, in clean words, with its date. Where he gave a formulation for the talk, it is
  reproduced as given.
- **(A)**: an agent's inference or proposal. It is not the authors' position until they confirm it.
- Nothing in the deck is approved unless the authors discussed it explicitly. A position is re-confirmed before a
  slide depends on it, and again when a related position changes.

## Setting

"Toward Active-Vision Foundation Models", MAIN 2026 (Montreal AI and Neuroscience), 2026-10-30: 25 minutes and 5 of
questions. An AI talk in a neuro-AI session; it is not about videogames. Part I by Sabrina Du, Part II and the
outro by Yohaï (Y, 2026-10-25: Sabrina likely presents Part I only, with no outro). The
paper (arXiv:2603.22570) is accepted at NeurIPS 2026; its camera-ready is due 2026-11-06.

## Positions

Stated 2026-10-24:
- (Y) Practicality, a principle of the project, runs through the talk: what works, not beautiful ideas that do not
  work. This will not change.
- (Y) "Before knowing where to look, we should know how to see." Prior active models did not yet see well, so it
  makes sense to fix the observer before trying to give them a great policy.
- (Y) The distinction between observer and policy is one of the work's most important conceptual moves and must be
  made explicit. Before it, the audience must understand what active vision is, what it has been about, and why they
  might care.
- (Y) "The observer is the bottleneck, not the policy" goes in, and the nuances of the argument must be preserved.
- (Y) Training observer and policy together, done naively, works badly, for reasons that stay separate (wording
  corrected 2026-10-24 late: co-adaptation itself is not what works badly):
  - Specificity. Observer and policy co-adapt. That can in principle give peak performance with that policy; the
    cost is an observer specific to its policy, and so inflexible (detailed below).
  - Instability. It comes from policy gradients, that is from training with reinforcement learning: gradient
    variance, instability, possibly collapse. Reinforcement learning should be avoided whenever possible.
  - Value. Why pay for policy-specific pretraining when learned policies are probably going to be rather basic
    (AdaGlimpse's Fig. 8, possibly shown whole on a slide). Reconsidered 2026-10-25: may not be relevant.
- (Y) Joint training remains possible later; a frozen observer with a policy on top is what this work did; a
  policy-agnostic observer can be an initialization for joint training ("qui peut le plus peut le moins").
- (Y) Stability and signal are fundamental. Transferring knowledge about the world into the model is what training
  does in general: every optimizer step teaches something.
- (Y) The work has two major contributions, both to be understood and conveyed: the architecture (what it solves
  and why) and the pretraining (where policy agnosticism and the learning signal matter).
- (Y) On cost, a principle of the project: one should not waste enormous resources on the wrong problem. The point
  is not that CanViT needs DINOv3: models like DINOv3 give task-agnostic representations, cost a great deal to train,
  are at the heart of much current research and have been applied in neuro-AI (Raugel et al.), and it is smart to
  use them. To discuss further.
- (Y) CanViT at 21 glimpses beside prior models at 8 is fair: they stopped at 8 because of diminishing returns; had
  they performed well, they would have shown it.
- (Y) A gain of 0.4 mIoU over EG-C2F is a big difference against a heuristic policy that is already excellent.
- (Y) CanViT's glimpses are uniform-resolution crops, not foveated; eccentricity-dependent acuity is part of what
  CanViT 2 covers. Reference frames are important and are discussed, as a functional argument: CanViT's backbone
  works in glimpse (retinotopic) coordinates and its canvas in scene (spatiotopic) coordinates. What is set aside for
  the time being is the neuroscientific plausibility of a spatiotopic representation (for instance against the
  predominantly eye-centered reference frame Xiao et al. 2024 found in ventral neurons). Clarified 2026-10-24 late;
  it replaces an agent's broader "avoid reference frames".

Stated 2026-10-24, reviewing the deck:
- (Y) The distinction between object classification and semantic segmentation, image-level (global) against
  pixel-level (dense) tasks, is important to get clear.
- (Y) The title slide names the speakers only, Yohaï and Sabrina. The paper gets its own slide, where the audience
  first hears "CanViT" and "Canvas Vision Transformer": the whole team, a QR code, and NeurIPS 2026. NeurIPS may also
  appear at the very end, with a screenshot and "see you in Paris!".
- (Y) Integrating visual information across viewpoints into a coherent understanding is part of the problem the architecture solves:
  part of understanding the key components of an active-vision model. It comes before the architecture.
- (Y) Where and how each result appears is still open; nothing is repeated. Peak performance and efficiency can be
  separated: efficiency at inference is a bonus, efficiency at training is what matters scientifically.
- (Y) The viewing policies are introduced before any result that uses them: no graph that means F2C or C2F before
  then. Pretraining policies and evaluation policies are introduced separately, on slides titled "Pretraining
  policies" and "Evaluation policies".
- (Y) A dedicated slide for the learned policy, as a closing touch: the talk argues against RL, then uses it at the
  end, efficiently. A possible title: "Learning where to look".
- (Y) "How to see" apart from "where to look" is an important distinction: policy-agnostic pretraining against policy
  learning.
- (Y) Pretraining's practical details: precomputing the teacher's features can be a divisive point; short training
  rollouts (truncated backpropagation through time) may confuse or be too technical, or may belong as part of the
  recipe, and the model generalizing to long sequences is a nice result; the paper holds the technical details.
- (Y) The learning signal is an important idea, but `#learning-signals` conveyed it badly and fit nowhere in the
  story; how and where to convey it is open (see information density, below).
- (Y) Showing how few glimpses learning where to look took, beside the pretraining's, as a reveal; RTX 4090 time can go
  with it. Built on `#learned-policy` 2026-10-24.
- (Y) `#pretraining`'s hours chart is not wrong: doing what DINOv3 did would take a long time. Its framing against the
  cost principle is a minor question for later.
- (Y) Whether the model is presented with custom animations or with the paper's figures, which may be better, is open.

Stated 2026-10-25 evening, reviewing the deck's structure:
- (Y) Policy-agnostic pretraining is an important point to get across; which arguments for it belong in the talk
  needs careful thought ("learned policies tend to be simple" may not be relevant).
- (Y) The parallel with LLMs (pretraining without RL, post-training with RL) may get a dedicated slide, after the
  learned-policy results.
- (Y) The results have two distinct angles, kept apart: the differences between viewing policies, and peak
  performance against existing models. The second, CanViT's peak ADE20K and ImageNet-1k results against prior
  active models, is missing from the deck; showing the narrowing gap with passive models, without a whole
  dedicated slide, possibly where the new active-vision state of the art is said, would be valuable. FLOP efficiency
  is not dealt with either; to think about.
- (Y) Canvas Attention stays. SR-RoPE must be explained or at least mentioned: it is how the model makes sense of
  space, fundamental to the work.
- (Y) Policy agnosticism makes replaying recorded scanpaths possible: worth getting across, even in a few seconds,
  plainly, without over-claiming or anchoring on papers and datasets the authors do not know well. Not settled.
  Uniform-acuity glimpses are not a limitation of policy agnosticism; no caveat about eye-centered reference frames.
- (Y) On the learned policy, "cheap" is always qualified: cheap to train, and it works well.
- (Y) `#compare`'s "Rich world knowledge" ✗ for prior active models can be softened to an intermediate mark
  (AdaGlimpse starts from MAE weights); not critical.

Stated 2026-10-25 late, the pretraining section's logic (supersedes earlier agent framings of it):
- (Y) The question: with the architecture in hand, how to pretrain it into a foundation model. It is first a
  representation learning problem, not an RL problem; RL brings noisy, high-variance gradients, a sparse learning
  signal, and observer and policy coupled.
- (Y) Problem 1: how to make the model robust to any policy. Answer: policy-agnostic pretraining. Behind it: task
  agnosticism, since where people look depends on the goal (Yarbus, a callback); a policy-specific observer co-adapts
  with its policy (even with a stop-gradient), with no guarantee under others; and, independently, joint training
  through policy gradients is unstable. Whether instability is said is open; the nuance is kept.
- (Y) Problem 2: how to give the model an understanding of the visual world, rich world knowledge, without an
  enormous amount of computation, costly and ecologically wasteful. Answer: dense latent distillation from DINOv3,
  which supplies that knowledge for a scene seen whole; what CanViT must learn is building it from glimpses.
- (Y) The label-free angle stays, at both levels: DINOv3 learned without labels, and CanViT learns from its features without
  labels (a callback to Part I's supervised against self-supervised).
- (Y) Payoffs come later: cheap policy training on the frozen observer (the learned-policy results); replaying
  recorded scanpaths (likely the ending).

Stated 2026-10-25 late, on the open threads:
- (Y) The pretraining section, so that people understand: the two problems to solve and the two answers; then answer
  1 in detail, policy-agnostic pretraining; then answer 2, dense latent distillation (where the learning-signal argument
  belongs); then the two together, the policy-agnostic dense latent distillation slide.
- (Y) The canvas's own moment may already be the SR-RoPE slide; not sure.
- (Y) The LLM parallel: possibly a quick animation on "Learning where to look", after the result. Built, then set aside the same day (Y: too verbose); said aloud while the policy-learning budgets are shown.
- (Y) "Extrapolation to unobserved regions" stays in Backup for now (the title slide's teaser shows extrapolation; it
  may be said aloud); a direct, streamlined callback to the table slide would be good: quick, not several rows and
  columns.
- (Y) No softening of "Rich world knowledge" for prior models for now. CanViT's red bars: low priority. AME's two
  names: does not matter. Segmentation palettes: unify if it can be done meaningfully.

Stated 2026-10-24, in a discussion between the speakers:
- (Y) Say "observer", never "vision encoder": an encoder covers instantaneous vision only and leaves out memory.
- (Y) Use the terms policy-specific and policy-agnostic pretraining. Training observer and policy at the same time
  (co-training) differs from training them jointly through shared gradients; a stop-gradient allows the first without
  the second. They raise different problems:
  - Policy-specific pretraining, even with a stop-gradient, makes observer and policy co-adapt: what the observer
    sees in pretraining is shaped by the policy, and the policy reads features of what the observer saw. The observer
    may then work well with its own policy and fail or underperform badly with others. It is inflexible: a new policy
    (scanpaths recorded from humans or monkeys, or a better policy found later) comes with no guarantee, and getting
    a new policy can take unnecessarily long when its training is coupled to the model's.
  - Instability and gradient variance come from policy gradients; with a stop-gradient, the observer does not
    receive them.
- (Y) Replaying recorded scanpaths through the observer is a strong argument for policy-agnostic pretraining: the
  model is built for any sequence of fixations, so human or monkey scanpaths can be tried with it meaningfully.
- (Y) Task-dependent scanpaths motivate policy agnosticism (Sabrina's suggestion): a foundation model should be
  task-agnostic, and Yarbus showed that viewing patterns depend on the viewer's goal, not only on the image; so the
  model should handle any sequence of fixations. Put Yarbus on screen; it is taught in introductory neuroscience and
  psychology, so much of the audience will know it.
- (Y) A flow for the talk, proposed in the discussion: what a foundation model is (much training, task-agnostic, a
  general-purpose understanding); passive vision has them, active vision does not; what active vision needs that
  passive vision does not (memory; retinotopic against spatiotopic coordinates); the architecture provides those
  components, as the paper lays out; then how to make it a foundation model: task agnosticism argues against pretraining
  the observer with a policy learned by RL, since scanpaths depend on the goal (Yarbus), which motivates policy agnosticism. Beyond that argument,
  sub-arguments: replaying scanpaths and not overfitting to a policy; and complexity removed, since pretraining comes
  first and policy training afterwards can be far cheaper. Show how quick post-hoc policy training is: the learned
  policy took roughly 600 thousand to a million glimpses, pretraining about a billion (exact count: the RL repository's
  results notes, private until it goes on a slide).
- (Y) The nuances and counterarguments stay recorded even when the talk does not say them all.
- (Y) The parallel with LLMs belongs in the talk: pretraining without RL, post-training with RL.
- (Y) The deeper point is information density and what the model learns, and policy agnosticism is a practical answer
  to it. RL serves to learn where to look, but knowing whether a look was good says little about the world; predicting
  what one will see teaches about the world. Active-vision papers mix the two objectives, one for the policy and one
  for the observer. RL is information-scarce compared with supervised or self-supervised learning; possibly an
  information-funnel graphic, possibly with studies of how much information RL post-training gives LLMs. RL is a last
  resort, not useless. Provocation, if any, is spoken; not necessarily a quote on screen.
- (Y) Be prepared for another meaning of "active-vision foundation model": a model given a task in language that
  selects where to look, as I-Perceive does (`sources/literature.md`); a possible future direction, from a different
  tradition (camera poses in 3D scenes).

Stated 2026-10-24 evening, in two discussions between the speakers:
- (Y) Vision foundation models must be introduced before Active-Vision Foundation Models; the talk introduces DINOv3
  specifically, since it returns later (Raugel et al.; CanViT's teacher). The point: image in, semantic features out,
  directly meaningful without adapting the model; that abstraction boundary is what makes its output usable as a
  training signal. No need to contrast with MAE or fine-tuning.
- (Y) PCA is signposted on the DINOv3 slide as how the talk shows features throughout.
- (Y) The key distinction on the models-of-vision slide is supervised against self-supervised: learning purely from
  visual data, without imposing a human understanding of the world through labels (CLIP is flexible but supervised by
  human pairings, so "flexibility" is not the argument). "General", not "flexible".
- (Y) Raugel et al.'s Fig. 5E, framed as a requirement, not a bonus: modeling higher regions appears to require more
  training (the half time grows with distance from V1). The question of how this fits the debate that bigger, better
  models are not more brain-like is open and worth reading on.
- (Y) What's missing from these models, said in one breath: gaze shifts and a partial, high-acuity view (the fovea);
  this ties back to CanViT 2's eccentricity-dependent acuity.
- (Y) The history of active vision in two acts: a few deep active computer vision models tracing back to RAM, then,
  more recently, neuro-AI active models; "drawing inspiration from this process".
- (Y) Titles might be removed from many slides in the end; to decide later.

Stated 2026-10-23 (Montreal time; from 2026-10-24 01:00 UTC on, this session's earliest messages):
- (Y) Ideas the talk needs, to think about critically: active vision; foundation models; why neuro-AI cares about
  foundation models; how to train them without breaking the bank; supervised against self-supervised learning against
  RL; practical issues such as instability. A point raised is not necessarily a slide; every important point is kept
  track of. Yes/no comparison tables are welcome for storytelling.
- (Y) The audience does not necessarily know that vision is active better than we do; going quickly over the logic of
  deep networks as models of the visual system is worth it, to show the logic of the motivation.
- (Y) The Yarbus figure likely will not speak as it is.

- (Y) "Action" means action selection: where and at what zoom to look next. Nothing broader is claimed.
- (Y) An AI talk about getting good foundation models for action-aware research: what belongs in the policy and what
  in the observer, policy-agnostic pretraining, avoiding RL when you can. The audience should leave with a recipe
  they can use.
- (Y) The learned-policy results (unpublished) are included.
- (Y) CanViT 2 appears in broad strokes only (eccentricity-dependent acuity, dynamic scenes); no details anywhere in
  this repository.
- (Y) Free-viewing neural datasets: a quick slide with figures only, which may be dropped.
- (Y) Lamarck Labs joins the funders on the last slide (the paper acknowledges its compute support).
- (Y) Timings are guesses until rehearsal, and agents' estimates are likely to be far off.

Stated 2026-10-01, for an earlier design of the talk; to re-confirm before relying on them:
- (Y) The logical order and the propositions each step proves matter most. How CanViT was built serves what it does;
  overall architecture and policy-agnostic pretraining first, Canvas Attention in Backup. (The order itself lives in
  one place, "The deck as it stands"; it can change at any time.)
- (Y) Efficiency only as a link (scale needs cheap training); for scientists, training and adaptation cost first,
  inference second.
- (Y) Never say CanViT saw "an order of magnitude more" data than previous active models (the paper's abstract, the
  README and the project page still say it; remove it for the camera-ready).
- (Y) The authors' faces
  round (on the title then; 2026-10-24: the speakers on the title, the whole team on `#paper`); QR codes large at the
  end; the live demo in Backup (removed 2026-10-24); control experiments not as slides; no
  remapping work from the Journal of Vision; unsettled slides go to Backup.

Reviewed by Yohaï 2026-10-24 and dropped from the list above: "compare like with like, never against far larger
models" (he does not recall saying it); "under 800 USD" for the ImageNet-1k fine-tuning (true, in
`sources/checked-facts.md`, but the talk will not mention it); a list of keywords to highlight (trans-saccadic
integration, boundary extension, metacognition, order matters, pixels as the identity teacher: mostly out of place or
not worth highlighting); a fixed opening order.

## Agent inferences and proposals

- (A) A summary of the talk, which Yohaï judged a fair start (2026-10-24): active vision has long promised vision
  that is more efficient and closer to biology but lagged far behind passive models; separating seeing from choosing
  where to look, and building a general observer the way foundation models are built, is a way forward; CanViT is a
  first step, which makes active-vision questions practical to study, including for neuro-AI.
- (A) How the two contributions fit, from paper §4–5 (Yohaï skimmed this account on 2026-10-24 and found it much
  better than before; not checked line by line): glimpses from arbitrary viewpoints cannot be carried forward as a
  grid, so CanViT keeps memory in a canvas fixed in scene coordinates; that makes dense outputs a token-wise readout
  and lets gradients flow across time. The canvas is large, so it has no self-attention or MLP and Canvas Attention
  projects only on the glimpse side. Pretraining asks, after every glimpse, for the teacher's features of the whole
  scene: spatially dense (unseen regions too, which demands integration and extrapolation) and temporally dense (a
  loss every step, so short truncated BPTT chunks with constant memory). A canvas fixed in scene coordinates matches
  the teacher's scene-wide patch grid token for token, which makes that target directly usable.
- (A) Prior work's own ablations (`sources/literature.md`) suggest that choosing where to look added about 0 to 3
  points on natural images with a low-resolution glance, many glimpses or dense segmentation, and 5 to 70 points
  where information is small, sparse and off-centre; and that within a paper, changes to the observer often
  outweighed changes to the policy. Yohaï found this too zoomed in for now; how to scope the claim on stage is open.
- (A) Scanpath replay, on `#neuro-ai` until that slide was removed on 2026-10-24 ("human fixations, given a scale, become viewpoints: a
  policy on the same observer"): because CanViT is policy-agnostic, recorded human or monkey scanpaths can be replayed
  through it and its memory read after every fixation, to compare with fixation-locked recordings such as the
  free-viewing datasets. (Y) 2026-10-24: worth keeping in mind, and later that day a strong argument for policy-agnostic
  pretraining (Positions); caveats as above (uniform crops, not foveated; the Xiao et al. contrast above).
- (A) What the dense learning signal buys the observer has not been worked out with the authors. An earlier agent
  claim, that density is why CanViT trains without RL, is wrong: it could be trained without RL without a dense
  signal.

## The paper in the talk

What the paper says, and where each point stands in the deck (A, 2026-10-25; read against the paper in full). "Said"
means in the notes only, which the authors have not reviewed. Each entry ends with an agent's judgment, for the
authors to confirm.

- Why active vision: efficient and biologically plausible perception (abstract). Biology on screen (foveation, Yarbus);
  efficiency nowhere. Needed only if the FLOP result is shown.
- Viewpoint = center and scale; a glimpse has fixed pixels, so scale trades coverage against detail (Preliminaries).
  Said on `#vit-on-a-glimpse`. Should be visible: every policy, and "where and at what zoom", depend on it.
- Glimpses do not align with the scene, so patch tokens cannot be carried across time (Preliminaries, §4). Shown by
  `#integrating-information-across-viewpoints`.
- Canvas: persistent, scene coordinates, token-wise readout, gradients across time; no MLP or self-attention on it
  (§4). On screen in the paper's figure only; the figure is small on `#architecture`.
- Canvas resolution set at inference (grid broadcast from one learned patch, §4); trained at 32², better at 64² (§6).
  Nowhere. A distinctive property; candidate for the results or Q&A.
- Recurrent CLS token: carried across glimpses, read out for classification (§4, §6). Said only. Yohaï (2026-10-25):
  likely explained quickly with the architecture diagram.
- SR-RoPE: positions in scene coordinates bind the retinotopic backbone and the spatiotopic canvas; zoom is carried by
  the spacing of patch positions (§4). Said only. Must be explained (Y, 2026-10-25).
- Canvas Attention, asymmetric (§4). On screen; stays (Y).
- VPE token: a non-essential affordance for future policy learning (§4, App.). Set aside; Q&A if the learned policy
  prompts it.
- Distillation in DINOv3 latent space rather than pixels (§5); teacher features precomputed. On screen
  (`#distillation`); pixels against latents only in Q&A.
- CLS target in the loss (§5) and its ablation: dense supervision also improves CLS reconstruction (App.). Not shown;
  goes with the recurrent CLS token if anywhere.
- R-IID and F-IID, scale density favoring small glimpses (§5). On screen. F-IID speeds convergence (App.): set aside.
- Truncated BPTT over 2-glimpse chunks, random rollout length averaging 4; generalizes to 21 (§5, §6). Said on
  `#distillation`. Yohaï (2026-10-24): the generalization is a nice result; the mechanism may be too technical.
- 166 H100-hours, 13.2 M scenes, 1 billion glimpses (§6). On screen.
- Frozen weights with linear probes, "to assess feature decodability"; fine-tuning only for ImageNet-1k peak (§6). Said
  in the notes; whether it needs a visible moment is open. (An agent's claim that it is what makes the results a
  foundation-model claim was wrong, Y 2026-10-25.)
- Peak against prior active models (Table 1: 45.9 against 27.6 mIoU; 84.5 fine-tuned against 82.2 top-1). Missing (Y).
- One glimpse: 38.5 mIoU, above every prior active model's best, with 20× fewer FLOPs than AME at 27.6; also above its
  input- or FLOP-matched DINOv3 teacher (abstract, §6, Fig. 4). Missing. The abstract's headline; candidate for the
  peak and efficiency results.
- C2F far above R-IID and F-IID on ADE20K; EG-C2F more efficient; ImageNet-1k plateaus early (§6). On screen.
- F2C still beats prior models: perception, not viewpoint selection, was the bottleneck (§6). On screen (the widening
  click) and said.
- C2F above F2C on identical viewpoints: the paper reads it as "the importance of contextually informed processing"
  (§6). On screen as a fact; say what the paper says.
- RFS: recurrence alone improves accuracy, then declines without new viewpoints (§6). Removed from the deck; Q&A.
- Fig. 4 (canvas resolution, mask size, frontier against DINOv3). Nowhere; Q&A or Backup.
- Ablations (App.): reads matter, BPTT matters, ViT-S, VPE least. Set aside; the no-reads ablation is the evidence if
  anyone asks whether reads matter.
- Canvas evolution within a glimpse: writes go from local to scene-wide (App. figure). Set aside.
- Limitations and future work (§7): static scenes (`#next`); forgetting; RL policy learning (now done, `#learned-policy`);
  depth expected; a passive teacher is needed (cost at high resolution, 19 TiB of features), active-to-active
  self-distillation; one model size, a dataset 100× smaller than DINOv3's. Only static scenes is on screen.

## Open questions for the authors

- Kept open until addressed or decided (Y, 2026-10-25):
  - The architecture: the canvas has no moment of its own (it appears in the paper's figure on `#architecture`, now
    600 px wide). Built 2026-10-25, for the authors to review: `#sr-rope` after `#architecture` (a glimpse's 8 x 8
    cells and their scene positions, the canvas's 32 x 32 cells, a smaller glimpse whose cells pack together), and a
    click on `#architecture` ringing the recurrent CLS token where it leaves one glimpse and enters the next.
  - The pretraining: why not pretrain the observer with its own policy is not argued on screen; the notes of
    `#pretraining-desiderata` carry it (AI draft). Rebuilt 2026-10-25 after Yohaï's review, then `#pretraining-desiderata` moved to Backup (Y: it says little; said aloud instead, a model that understands the world and works with any policy): was `#pretraining-desiderata`
    (robust to any viewing policy; rich knowledge of the visual world without enormous compute; the answers on a click),
    then `#distillation` ("Policy-agnostic dense latent distillation"), `#pretraining-policies`, `#pretraining`, the
    table callback. `#how-to-see-first` is in Backup; where it belongs is open. The problem signposts and
    `#dense-latent-distillation` were removed (rejected).
  - The results (Y, 2026-10-25): Table 1's numbers side by side, as bars on a dedicated slide, could be interesting;
    each concept needs a clear home without messy repetition, and covering the same results from different angles is
    fine when the talk says so. Since 2026-10-25 the ImageNet-1k panel of "Accuracy by viewing policy" shows the
    fine-tuned curves (solid) beside the frozen ones (dashed), as the paper's Fig. 3C does.
  - The results: peak performance against prior active models is partly there (the ADE20K panel widening to AME and
    AdaGlimpse); the narrowing of the gap with passive models is not. Yohaï is thinking about how to present it.
    Candidates from the paper (A): Fig. 3A (accuracy against cumulative FLOPs, CanViT's curves far above the prior
    models' points; `#frontier` in Backup) and Fig. 4C (the Pareto frontier against DINOv3 ViT-B at input resolutions
    up to 512 px: CanViT above it at low FLOPs, close to its 47.2 at the top). Fig. 4C compares CanViT with its own
    teacher's architecture, which is the comparison the gap is about.
- Prior models on "Accuracy by viewing policy" (A, 2026-10-25): the ADE20K panel has AME and AdaGlimpse as lines
  (agreed 2026-10-24), the ImageNet-1k panel none, because only the ADE20K export carries baselines (ImageNet-1k's are
  typed in the paper's Table 1). Adding them there would mislead: AdaptiveNN's 82.2 sits above CanViT's frozen 81.1,
  and CanViT's 84.5 is fine-tuned and off that panel. The lines also span every glimpse count though prior results
  are at 8 glimpses. Proposal: the policy slide compares policies only; prior models, on both tasks with each
  number's protocol, go where peak performance is shown (Y, 2026-10-25: two angles kept apart).
- How Part I presents active vision as a field (what it has been about, why one might care) before the observer and
  policy distinction.
- How to word the functional argument for neuro-AI.
- The cost slide's framing (reuse of an investment, not a cheaper pretraining), to discuss.
- `#free-viewing` ("Toward AVFMs for Neuro-AI") is in Backup (Y, 2026-10-25): its precise argument must be thought
  through before it returns, without mistakes; one option is the Active Visual Semantics (AVS) dataset alone, named on
  screen with its figure. Whether to address LookWhere in the talk or only if asked.
- Slides with value but undecided (Yohaï, 2026-10-24): `#human-vision` (likely redundant, since the speaker can say
  that we move our eyes, and it shows foveation); `#extrapolation` (may be too heavy, may be reworked); `#gap-canvit`
  (repeats `#gap`'s title; putting results in context is wanted, but where is open); `#timeline` (has issues and its
  place is open; not to be deleted); a callback to `#compare` with CanViT's column animated in (not convinced it is worth it).
- A David Silver remark about AlphaFold (that the team should stop treating it as an RL problem), which Yohaï found:
  its source is not yet verified (a web search on 2026-10-24 did not find it). Needed before it is quoted anywhere.
- On-screen wordings an agent wrote that state more than the evidence may support: `#active-vision-model`'s "prior
  work focused here" (the paper: "often focused") and "A general-purpose observer lets you use any policy" (tested
  policies); `#compare`'s rows "Any viewing policy, no RL" and "Rich world knowledge" (Y 2026-10-24; was "Accuracy close to passive models") (45.9 against 47.2,
  at 21 glimpses).
- From 2026-10-01, still undecided: `#architecture` as the click-by-click drawing (the paper's full figure is in
  Backup as `#architecture-paper`); one segmentation
  palette for the talk; policy colors: settled 2026-10-25, the paper's (Y); "runs everywhere" (JAX and MLX ports) or PyTorch and the browser only.

## The deck: a design (agent's proposal, 2026-10-25; not approved)

Rethought from the positions above and the paper (read in full on 2026-10-25), after Yohaï's review found the deck
sloppy: the demo came before the architecture and pretraining were understood, signposting was missing, and
concepts were smuggled in. Principles: every move of the argument is marked, by a short signposting slide where
needed (a few words, 5 to 10 seconds); nothing the argument relies on is smuggled; each result appears once; the
demo comes last, when everything in it has been explained. Kinds: **S** a signposting slide, **C** a content slide.
Built on 2026-10-25 in this order (slide ids: `#passive-models`, `#how-to-see-first`, `#outline-architecture` with a
click that highlights the architecture, `#glimpses-do-not-line-up`, `#outline-pretraining`,
`#how-to-see-apart-from-where-to-look`, `#classification-and-segmentation`); the optional signposts 19 and 27 were
not built, their points said on `#architecture` and in the results' notes; `#gap-canvit` was removed, its numbers
said on `#cost`. Status words below are those of the proposal. Durations are guesses (Part I about 8 minutes, Part II
about 16, 1 of slack).

### Part I (Sabrina): why Active-Vision Foundation Models

1. C `#title` · 0:15 · exists. The title, the speakers and their affiliations on the left, on the brand gradient; on
   the right a silent teaser, CanViT running in a loop (Y, 2026-10-25), five glimpses per scene, each chosen to change
   the canvas a lot (Y). Scenes ADE_val_00000419 (a poplar in fields), 00001882 (tires and a treeline), 00000203 (a
   skyscraper), 00001860 (a street with a parked car), chosen by eye (A, 2026-10-25) among the 10 of 67 screened scenes
   whose glimpses 2 to 5 change the canvas most (`experiments/title_teaser`). Excluded: ADE_val_00001980, a paper
   machine (Y, 2026-10-25: looks odd).
2. C "Computational models of biological visual processing" · 1:15 · changed 2026-10-25. Yamins 2014 under
   "Supervised", then Zhuang et al. 2021 (Fig. 2B, IT) under "Unsupervised / self-supervised", each word large (Y:
   the key distinction of the slide). Raugel's DINOv3 result is said on `#foundation`.
   (Y) Approved for now, 2026-10-25.
3. C "DINOv3" · 1:00 · exists. Image and features side by side from the start (Sabrina, 2026-10-25), then the linear
   decoding.
3a. C "DINOv3 and the brain" (`#dinov3-and-the-brain`; its own slide since 2026-10-25, Sabrina: it was `#foundation`'s
   last click): Raugel et al.'s
   Fig. 5E (arXiv numbering), each cortical region's training half time against its distance from V1. (Y) The scaling
   argument: representations that match higher cortex take training at scale; that is what a foundation model is,
   why neuro-AI cares, and why the pretraining builds on DINOv3 instead of repeating its training. Says: trained once,
   without labels, at great cost; reused across tasks; also what neuro-AI now compares with the brain.
   (Y, 2026-10-25) The slide does not open on scale: it opens on the study, the paper's OpenReview header above its
   Fig. 1 (what Raugel et al. did: 7-Tesla fMRI and MEG to disentangle the factors of convergence between brains and
   DINOv3), and Fig. 5E follows on a click.
4. S "What's missing?" · 0:10. The implicit answer is eye movements, answered by Yarbus next (Y, 2026-10-25: the
   answer is not that passive models see the whole image, which feeding them a crop would disprove).
4b. C `#human-vision`, no title · 0:20 (from Backup, Y 2026-10-25): (Y, 2026-10-25) opens on the foveated glimpses, a white
   fixation cross jumping between fixations, then a click stops on the street at full resolution, the cross at its
   center, as the models see it; the foveation animation on the street,
   the answer to "What's missing?".
6. C "A brief history of deep active computer vision" · 1:00 · exists. Opens on RAM alone on a short axis, beside its Fig. 1A (glimpse sensor) and a Fig. 3 glimpse path; a click fades them, grows the axis and brings the later models in turn, without further clicks; then the neuro-AI models (Sabrina, 2026-10-25). Models since 2014; how each chose where to
   look (mostly RL); neuro-AI active models, each built for a task or a way of looking.
7. C "The wide gap between passive and active computer vision" · 0:45 · exists. ADE20K segmentation, then the bars.
8. C "What's in an active-vision model?" · 1:30 · exists. Clicks (Sabrina, 2026-10-25): the claim; then a teal contour around action selection (the focus of most prior active vision work), then, on the next click, a blue one around instantaneous vision (the focus of passive vision foundation models), in place of labels. The prior models' ablations left the main slide for its Backup copy, `#active-vision-model-ablations` (Sabrina, 2026-10-25). The three axes, observer and policy; "prior work often
   focused here"; prior papers' own ablations; "No policy can make up for a poor observer." The gap is the question,
   this slide a first answer.
9b. C "Where people look depends on their goal" · 1:00 (a click goes from the painting straight to all seven records, Sabrina, 2026-10-25) (moved 2026-10-25, Y: before the requirements of an
   AVFM). Yarbus: a model for any task must handle any sequence of fixations.
10. C "Toward Active-Vision Foundation Models" · 1:15 · exists. The requirements, a row per click, then the goal
    column.

### Part II (Yohaï): how, and what it does

11. C `#paper` · 0:30 · exists. Canvas Vision Transformer, CanViT, the team, NeurIPS 2026, QR code.
12. S "Architecture" and "Pretraining" (no article, Y) · 0:10 · new. Two words side by side; shown again at
    13 and 20 with the current one highlighted.

The architecture
13. S the contributions slide, "architecture" highlighted · 0:05 · new (a state of 12, or a copy).
14. C "Processing a glimpse with a Vision Transformer" · 0:45 · exists. Scene, glimpse, ViT, patch tokens.
15. C "Integrating visual information across viewpoints" · 1:00 · exists. Two glimpses of the table, a passive model
    on each, the middle unknown.
17. C "The canvas" (moved to Backup 2026-10-25, Y: out of the way for now; the signpost before it removed) · 1:15 · exists (built 2026-10-24, to review). Scene and canvas side by side; glimpse 1 writes;
    glimpse 2 reads, then writes; the rest played.
18. C "Canvas Vision Transformer architecture" · 1:15 · change. The paper's figure (`arch_overview.svg`), revealed
    in parts with masks or crops that assemble: the backbone on a glimpse; the canvas; reads and writes alternating
    across depth; the next glimpse. Says: retinotopic backbone, spatiotopic canvas, scene-relative RoPE binding
    them; no MLP or self-attention on the canvas, so it is cheap and can be large. Replaces the hand-drawn version.
18b. C "Canvas Attention" · 0:45 (from Backup 2026-10-25, Y): the paper's Figure 3A; the asymmetric read and write,
    why the canvas is cheap and can be large.
19. S "Retinotopic backbone, spatiotopic canvas" · 0:10 · optional, new: the reference-frame point on its own (Y:
    reference frames discussed functionally).

The pretraining
20. S the contributions slide, "pretraining" highlighted · 0:05 · new.
20a. S "Before knowing where to look, we should know how to see." · 0:10 (moved 2026-10-25, Y: the case for
    policy-agnostic pretraining, not Part I).
21. C "AdaGlimpse's learned viewing policies" (to Backup 2026-10-25, Y: out of the way for now) · 1:00 · exists. Fig. 8 whole. Says: training an observer with its
    own learned policy; co-adaptation makes it specific to that policy; policy gradients bring instability; and the
    learned policy ends up nearly fixed.
22. C "Learning signal per glimpse" (moved to Backup 2026-10-25, Y: rejected as drawn; how to convey information
    density stays open) · 0:45. Deleted 2026-10-25: it measured information by counting numbers (a reward's one
    against the teacher's 787,200), which is meaningless: pixels give as many numbers and teach far less.
    Says: a reward says whether a look was good, not what is in the scene; LLMs: pretraining without RL, RL after.
24. C "Pretraining policies" · 0:45 · exists (before distillation, agreed 2026-10-24). R-IID and F-IID, static: each
    policy's rollout on a scene with numbered boxes, then its glimpses (Y: static, timesteps, prominent crops).
25. C "Passive-to-active dense latent distillation" · 1:15 · exists. The training step unrolled over glimpses.
26. C "Pretrained in 166 hours on a single H100" · 0:30 · exists. Against DINOv3 7B's 61,440 H100-hours.

What it does
27. S "What it does, frozen" (wording to find) · 0:05 · new, optional: results use frozen weights and a linear readout.
28. C "Extrapolation to unobserved regions" (to Backup 2026-10-25, Y: feels orphaned and unpolished, its place
    unclear) · 1:00 · exists. The table again, answered.
29. C "Coarse-to-Fine (C2F) and Fine-to-Coarse (F2C)" · 0:30 · exists. Static, in stages (Y, 2026-10-25: the
    animations side by side hid the difference): each policy a row of its quadtree levels in order, the boxes of each
    level on the street with its glimpse range (Y, 2026-10-24: R-IID and F-IID are not introduced again).
29a. C "Entropy-Guided Coarse-to-Fine (EG-C2F)" · 1:00 · exists (was "Uncertainty-based viewpoint selection", after
    the results). How EG-C2F chooses. Every policy is introduced before the results (Y, 2026-10-24).
30. C "Image classification and semantic segmentation" (moved to Backup 2026-10-25, Y: not needed; ADE20K is explained
    on #gap) · 0:30 · new. A label for the whole image (the dog) against
    a class for every pixel (the room): the second task, introduced on its own before its first result.
31. C "Accuracy by viewing policy" · 1:00 · exists. ImageNet-1k, then ADE20K. Says: works under every policy tested;
    order matters (C2F against F2C); even F2C beats the prior models: perception, not viewpoint selection, was the
    bottleneck (paper), the answer to slide 8.
33. C "Training cost" (moved to Backup 2026-10-25, Y: rejected as drawn; how to show training cost and where
    CanViT meets prior models is open) · 0:45 · change. Peak accuracy against each model's own training compute, both tasks; this is
    where CanViT is compared with prior models (`#gap-canvit` removed: it repeated `#gap`); the thumbnails on the
    headings go (slide 30 introduces classification).
34. C "Learning where to look" · 1:15 · exists. The learned policy over the baselines, then the glimpse budgets.
35. C "CanViT in action" (to Backup 2026-10-25, Y: the title slide carries a silent demo) · 0:45 · move to here. The full demo, now that every part of it has been explained.

Outro
35a. C "Quickstart" · 0:30 · exists (from Backup, Y 2026-10-24: right after the demo). `uv add canvit-pytorch`,
    then the code walked through line by line, its result beside it.
36. C "Toward CanViT 2" (title, Y 2026-10-24) · 0:30 (rebuilt 2026-10-25, Y): CanViT's column (static scenes, uniform acuity, in red) centered,
    then it shifts, an arrow draws, CanViT 2's column (dynamic scenes, eccentricity-dependent acuity, in green) appears.
37. C "Toward AVFMs for Neuro-AI" (title, Y 2026-10-25: active-vision neuro-AI already exists; to Backup the same day
    until its argument is settled, see the open questions) · 0:30 · the outro to neuro-AI; replaying recorded scanpaths
    through a policy-agnostic observer (Y), said and shown.
38. C `#links` · stays up during questions.

Backup: inference compute (`#frontier`), `#human-vision`, `#history`, `#results`, `#detail`,
`#canvas-attention`, `#architecture-paper`.

### Not yet on screen, from the positions

Now said in the notes but with no visual of their own: replaying recorded scanpaths; the two contributions named as
such (end of `#architecture`); "before knowing where to look, we should know how to see" (`#active-vision-model`);
the LLM pretraining/post-training parallel; why neuro-AI cares about foundation models
(`#foundation`); the complexity that policy-agnostic pretraining removes. Not anywhere yet: a slide that introduces
classification (tied to where the ImageNet-1k results go).

Removed on 2026-10-24 after Yohaï's review (git history has them): `#compare-canvit`, `#neuro-ai`, `#avfm`,
`#teacher`, `#calibration`, `#live`, `#learning-signals` (labels against rewards against distillation targets),
`#memory` (memory kept against reset on a street; true, but it served nothing); the scale facts under
`#foundation`; the practical-details panel of `#pretraining`; the setup list beside `#learned-policy`'s chart.

## Open work

- Rehearsal timings (run-throughs; agents' estimates are guesses until these exist):
  - 2026-10-25, Sabrina with Yohaï: `#title` 45 s, more than the notes say; can come down if needed.
  - 2026-10-25: `#tradition` 50 s (retitled "Deep neural networks as models of vision", Y).
  - 2026-10-25: `#foundation` 45 s.
  - 2026-10-25: `#dinov3-and-the-brain` 35 s.
  - 2026-10-25: "What's missing?" (`#passive-models`) 10 s.
  - 2026-10-25: `#human-vision` 15 s.
  - 2026-10-25: `#timeline` 35 s.
- Yohaï's feedback on Sabrina's run-through of Part I (2026-10-25), not yet applied (to be worked through with her):
  - Title: the contrast with passive computer vision models is said too loosely.
  - `#tradition`: "a new class of models" is questionable (new?); spell out IT; say where the better categorization
    performance predicts neural responses (which recordings).
  - Self-supervision is a relaxation of constraints, not "more biologically plausible constraints".
  - `#foundation`: spell out principal components, three at a time, before "PCs"; say who "we" is.
  - Scale is not a reason to care about DINOv3 in itself: it is the means to the semantic representation, which
    requires it.
  - `#dinov3-and-the-brain`: too quick; say what Raugel et al. did and what the paper is about.
  - "What's missing?" and `#human-vision`: the non-uniform front end (foveation, eccentricity-dependent acuity) was
    glossed over.
  - `#timeline`: the neuro-AI active models are passed over too quickly.
  - `#gap`: introduce ADE20K segmentation first: what the task is, why it is interesting (harder than classification,
    both spatial and semantic).
  - `#active-vision-model`: too much text, partly redundant; remove the descriptions under each component.
  - "No policy can make up for a poor observer" needs its reasoning said, not just asserted.
- `#rollout`'s scene and R-IID seed chosen so that its mIoU never drops (Y), by `experiments/rollout_demo`, if
  `#rollout` returns from Backup (parked 2026-10-25, Y).
- The meter on `#rollout` shows a scene's mIoU (A, 2026-10-25): about 18 to 30% on the street, against the benchmark's
  45.9, because one scene's mIoU counts every class it predicts there, rare ones included, while the benchmark pools
  over the validation set. Whether that number on screen helps or confuses is for the authors.
- `#compare`'s "Rich world knowledge" puts ✗ for prior active models (A, 2026-10-25), though AdaGlimpse starts from
  MAE weights pretrained on ImageNet; an expert may object.
- The pretraining chain (Y 2026-10-24: the AdaGlimpse slide leading to grayed-out "How to see" / "Where to look"
  boxes was odd): reworked by an agent as the principle (`#how-to-see-first`), then `#pretraining-policies` (AdaGlimpse's
  learned policies went to Backup, 2026-10-25), whose notes say the disentangling; the boxes slide removed. For the authors to review.
- Agreed 2026-10-24, not yet applied: AME (27.6) and AdaGlimpse (25.7) as reference lines on the ADE20K panel of
  "Accuracy by viewing policy".
- `#rollout`'s meter shows pixel accuracy, never introduced; mIoU instead (Y, 2026-10-24). The DINOv3 whole-scene
  panel is off `#rollout` for now (Y, 2026-10-24); `experiments/whole_scene_baseline` still computes its data.
- RFS is off "Accuracy by viewing policy" (A, 2026-10-24) because no slide introduces it; the authors decide whether it
  comes back with an introduction.
- Asked 2026-10-24 and never run (the discussion moved on): how much AdaGlimpse's released reconstruction model loses
  when it gets glimpses its policy would not choose, with confidence intervals. Its encoder ignores glimpse order
  (`sources/literature.md`), so swapping the quadrants' order alone would change nothing after the last glimpse; an
  agent proposed shifted quadrants, random positions at the same scale, and the classification policy's glimpses.
- From an agent's audit of the whole deck (2026-10-24 late), not yet decided or fixed:
  - Classification is introduced where ImageNet-1k first appears, on `#cost` (a dog with its label beside the chart
    heading); whether that is where the ImageNet-1k results belong stays open (Y).
  - "Toward Active-Vision Foundation Models" is on screen three times: `#title`, `#compare`, `#paper`.
  - Glimpses count from 1 throughout the talk ("glimpse 1"), as on its charts; the components show the paper's t
    elsewhere (`data-glimpse-label` on `<html>`).
  - Segmentation palettes differ between slides; on `#extrapolation` floor and ceiling now take a brown and a pale
    yellow, apart from the glimpse boxes' blue (experiments/table_corners/plot.py).
  - GPN: Sabrina's outline cites "Thorat et al. 2026, Science Advances"; a search found only arXiv:2511.12715 (2025),
    which the deck cites.
- `#free-viewing` says replaying recorded scanpaths through the observer (notes, 2026-10-25) but does not show it:
  showing it needs a recorded scanpath turned into viewpoints and run through CanViT-B (a scanpath from one of the
  free-viewing datasets, or Yarbus's records digitized); none is in the repository.
- From an independent review (a fresh agent, 2026-10-25), for the authors to decide:
  - The demo stays last, after `#learned-policy`: Yohaï suggested it could come at the very end (2026-10-24).
  - "Pretrained in 166 hours on a single H100" is the paper's claim as a title; "CanViT 2" and "Accuracy against
    training compute" replaced a colon title and an understated one (2026-10-25).
  - `#cost`'s on-screen conditions line ("Not counted: …"), which the rules would send to the notes.
  - `#active-vision-model`'s ablations now show the conditions they were measured in (after a whole-image glance;
    8 glimpses), and the notes give the larger gain where what matters is small and off-centre (GFNet on traffic
    signs, 69 to 91.5%). Whether to show more rows is the authors' call (2026-10-25).
  - `#classification-and-segmentation` shows `#gap`'s living room again beside the dog.
  - `#distillation` shows the released model over 21 glimpses (training averages 4), and its first glimpse is a
    nearly blank wall.
  - Length: about 35 main slides, several of them signposts of 5 to 10 seconds. Cuts wait for the rehearsal,
    since timings are guesses until then (Y); candidates: `#timeline`'s neuro-AI step, `#extrapolation`.
- Sabrina reworks Part I over the weekend of 2026-10-24; merge her version.
- Sabrina, 2026-10-25: write the keywords and spoken text for every slide before `#paper`, slide by slide in
  discussion with her, she writing the text and the slide following it.
  - Her idea (2026-10-25), undecided: before `#tradition`, a short slide on the first deep networks used as models
    (Yamins et al. 2014's) and the images they were tested on.
- Missing material: a moving-scene clip for `#next`.
- Known defects: segmentation colors differ between Backup slides (the components, on `#rollout`, use the ADE20K
  palette, `#extrapolation` canvit-pytorch's `LABEL_COLORS`); the main deck's maps share experiments/segmentation_colors.py
  (checked 2026-10-25); amber means every prior model on some
  slides and AME alone on `#timeline`.
- For the paper's camera-ready (2026-11-06), not the talk: prepared on the local branch `paper-camera-ready-fixes`
  (from `main`, not pushed, paper builds with no Type 3 fonts): AME's authors with their diacritics, AdaGlimpse dated
  2024, Raugel et al. dated 2026 (its ICLR edition), and the "order of magnitude more" clause removed from the
  abstract of the paper, the README and the project page. Left open: GAE's pages (16137–16146 in the bib; IEEE's
  record gives 16117–16126, possibly a CVF/IEEE pagination difference) and DINOv3's month (`feb` on OpenReview, the
  TMLR PDF header says 04/2026).
